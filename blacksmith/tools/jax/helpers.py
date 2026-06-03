# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from blacksmith.tools.logging_manager import TrainingLogger


def build_schedule(learning_rate, warmup_ratio, num_train_steps: int):
    warmup_steps = int(warmup_ratio * num_train_steps)
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, learning_rate, warmup_steps),
            optax.linear_schedule(learning_rate, 0.0, num_train_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )
    return schedule


def kl_divergence(p_logits, q_logits, T):
    p = nn.softmax(p_logits / T, axis=-1)
    log_p = jax.nn.log_softmax(p_logits / T, axis=-1)
    log_q = jax.nn.log_softmax(q_logits / T, axis=-1)
    kl = jnp.sum(p * (log_p - log_q), axis=-1)
    return (T**2) * jnp.mean(kl)


def ce_with_labels(logits, labels):
    num_classes = logits.shape[-1]
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    return optax.softmax_cross_entropy(logits, one_hot_labels).mean()


def cosine_embedding_loss(x, y, eps=1e-8):
    x_norm = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)
    y_norm = y / (jnp.linalg.norm(y, axis=-1, keepdims=True) + eps)
    cos_sim = jnp.sum(x_norm * y_norm, axis=-1)
    return 1.0 - jnp.mean(cos_sim)


_LOG_EPS = 1e-12


def clamped_softmax_cross_entropy_per_token(
    logits_f32: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    row_max = jax.lax.stop_gradient(jnp.max(logits_f32, axis=-1, keepdims=True))
    shifted = logits_f32 - row_max
    log_normalizer = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1))
    one_hot = jax.nn.one_hot(labels, logits_f32.shape[-1], dtype=logits_f32.dtype)
    picked = jnp.sum(one_hot * shifted, axis=-1)
    return log_normalizer - picked


IGNORED_LABEL = -100


def masked_cross_entropy(
    logits: jax.Array,
    labels: jax.Array,
    *,
    ignored_index: int = IGNORED_LABEL,
    clamped: bool = True,
) -> jax.Array:
    """Shift-by-one causal cross-entropy with label masking.

    Positions where labels == ignored_index are excluded from the mean.
    When clamped is True the TT-safe integer-label CE variant is used (no
    one-hot, no softmax tensor); otherwise plain optax integer-label CE.

    Args:
        logits: (batch, seq_len, vocab) model output.
        labels: (batch, seq_len) integer labels.
        ignored_index: Value treated as "don't care".
        clamped: Use the TT bf16-safe CE variant.
    """
    shift_logits = logits[:, :-1, :].astype(jnp.float32)
    shift_labels = labels[:, 1:].astype(jnp.int32)

    valid = shift_labels != ignored_index
    safe = jnp.where(valid, shift_labels, 0)

    if clamped:
        per_token = clamped_softmax_cross_entropy_per_token(shift_logits, safe)
    else:
        per_token = optax.softmax_cross_entropy_with_integer_labels(shift_logits, safe)

    masked = per_token * valid
    return jnp.sum(masked) / jnp.maximum(jnp.sum(valid), 1)


def vocab_parallel_cross_entropy(
    logits: jax.Array,
    labels: jax.Array,
    mesh: Mesh,
    *,
    model_axis: str = "model",
    data_axis: str = "data",
    ignored_index: int = IGNORED_LABEL,
) -> jax.Array:
    """Causal cross-entropy for a vocab(model)-sharded logits tensor.

    A sharding constraint replicates the vocab axis (an all-gather over
    model_axis, which TT supports), making the softmax normaliser a local
    reduction so the standard clamped CE can be reused. This avoids the
    shard_map manual-computation region TT only allows once per module.
    """
    data_spec = data_axis if data_axis in mesh.axis_names else None

    # Replicate the model-sharded vocab axis (all-gather over model_axis) so the
    # cross-entropy reductions are entirely local; no shard_map / manual region.
    logits = jax.lax.with_sharding_constraint(logits, NamedSharding(mesh, PartitionSpec(data_spec, None, None)))

    return masked_cross_entropy(logits, labels, clamped=True, ignored_index=ignored_index)


def _vocab_parallel_embedding_lookup_local(
    local_embedding: jax.Array,
    local_input_ids: jax.Array,
    local_vocab_ids: jax.Array,
    model_axis: str,
) -> jax.Array:
    """Embed one vocab shard's tokens, run inside a shard_map.

    Each shard one-hots the vocab ids it owns and matmuls with its rows; only
    the owning shard contributes a non-zero row, so a psum over model_axis
    reconstructs the full embedding. Uses only matmul + psum (no row gather),
    since the row-sharded jnp.take is not legalizable on TT.

    Args:
        local_embedding: This shard's embedding rows, shape [v_local, hidden].
        local_input_ids: Token ids, shape [b, s] (replicated over model).
        local_vocab_ids: Global vocab ids this shard owns, shape [v_local].
        model_axis: Mesh axis the vocab dim is sharded over.

    Returns:
        Embedded tokens, shape [b, s, hidden], identical on every shard.
    """
    # [b, s, v_local]: True only where a token id belongs to this shard's slice.
    match = local_input_ids[..., None] == local_vocab_ids[None, None, :]
    one_hot = match.astype(local_embedding.dtype)

    # [b, s, v_local] @ [v_local, hidden] -> [b, s, hidden]. Shards that do not
    # own a token contribute all-zero rows, so the cross-shard sum selects the
    # correct embedding row. A one-hot matmul avoids the row gather (jnp.take).
    local_embeds = jnp.matmul(one_hot, local_embedding)
    return jax.lax.psum(local_embeds, model_axis)


def show_predictions(
    collected: list[dict],
    tokenizer,
    *,
    num_tokens: int = 20,
    ignored_index: int = IGNORED_LABEL,
    training_logger: Optional[TrainingLogger] = None,
) -> None:
    """Print collected prediction examples (CPU-only, no forward pass).

    Each dict in collected must have input_ids and labels.  Optional
    keys: predictions, per_token_loss, loss.  The EasyDel JAX eval JIT
    emits (loss, predictions) from a single fixed-signature program, so
    predictions are typically populated; per_token_loss is omitted on TT
    multi-chip because expanding the eval JIT output further has not
    been validated under the mesh-reopen workaround (tt-xla#1993,
    tt-xla#4809, tt-mlir#3963).
    """
    log = training_logger.info if training_logger is not None else logging.getLogger(__name__).info

    for i, ex in enumerate(collected):
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        predictions = ex.get("predictions")
        per_token_loss = ex.get("per_token_loss")
        batch_loss = ex.get("loss")

        shift_labels = labels[1:].astype(np.int32)
        valid_mask = shift_labels != ignored_index

        log(f"\n--- Example {i + 1} ---")

        if not valid_mask.any():
            log(f"  (no unmasked target tokens; all labels == {ignored_index})")
            if batch_loss is not None:
                log(f"  Batch loss:   {batch_loss:.4f}")
            continue

        prompt_end = int(np.argmax(valid_mask)) + 1
        prompt_ids = input_ids[:prompt_end]

        valid_targets = shift_labels[valid_mask]
        target_ids = valid_targets[:num_tokens]

        input_text = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=True)
        target_text = tokenizer.decode(target_ids.tolist(), skip_special_tokens=False)

        log(f"  Input:        {input_text!r}")
        log(f"  Target IDs:   {target_ids.tolist()}")
        log(f"  Target text:  {target_text!r}")

        if predictions is not None:
            valid_preds_all = predictions[valid_mask]
            pred_ids = valid_preds_all[:num_tokens]
            pred_text = tokenizer.decode(pred_ids.tolist(), skip_special_tokens=False)
            log(f"  Pred IDs:     {pred_ids.tolist()}")
            log(f"  Pred text:    {pred_text!r}")

            correct = int((valid_preds_all == valid_targets).sum())
            total = int(valid_targets.size)
            acc = correct / max(total, 1)
            log(f"  Accuracy:     {correct}/{total} = {acc:.3f}")

        if per_token_loss is not None:
            valid_token_losses = per_token_loss[valid_mask][:num_tokens]
            log(f"  Token losses: {np.round(valid_token_losses, 4).tolist()}")
            mean_loss = float(per_token_loss[valid_mask].mean())
            log(f"  Mean loss:    {mean_loss:.4f}")
        elif batch_loss is not None:
            log(f"  Batch loss:   {batch_loss:.4f}")
