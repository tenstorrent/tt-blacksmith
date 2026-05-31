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
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

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
    one_hot: jax.Array,
    eps: float = _LOG_EPS,
) -> jax.Array:
    """Per-token cross-entropy robust to TT bf16 fused-softmax drift.

    Computes softmax, clamps to [0, 1], renormalises, then returns
    -sum(one_hot * log(probs)) per token.
    """
    probs = jax.nn.softmax(logits_f32, axis=-1)
    probs = jnp.clip(probs, 0.0, 1.0)
    row_sum = jnp.sum(probs, axis=-1, keepdims=True)
    probs = probs / jnp.maximum(row_sum, eps)
    log_probs = jnp.log(jnp.maximum(probs, eps))
    return -jnp.sum(one_hot * log_probs, axis=-1)


IGNORED_LABEL = -100


def masked_cross_entropy(
    logits: jax.Array,
    labels: jax.Array,
    *,
    ignored_index: int = IGNORED_LABEL,
    clamped: bool = True,
    vocab_size: int | None = None,
) -> jax.Array:
    """Shift-by-one causal cross-entropy with label masking.

    Positions where labels == ignored_index are excluded from the mean.
    When clamped is True the TT-safe CE variant is used; otherwise
    plain optax softmax cross-entropy.

    Args:
        logits: (batch, seq_len, vocab) model output.
        labels: (batch, seq_len) integer labels.
        ignored_index: Value treated as "don't care".
        clamped: Use the TT bf16-safe CE variant.
        vocab_size: Vocabulary size; inferred from logits when None.
    """
    shift_logits = logits[:, :-1, :].astype(jnp.float32)
    shift_labels = labels[:, 1:].astype(jnp.int32)

    v = vocab_size or shift_logits.shape[-1]
    valid = shift_labels != ignored_index
    safe = jnp.where(valid, shift_labels, 0)
    one_hot = jax.nn.one_hot(safe, v).astype(jnp.float32)

    if clamped:
        per_token = clamped_softmax_cross_entropy_per_token(shift_logits, one_hot)
    else:
        per_token = optax.softmax_cross_entropy(shift_logits, one_hot)

    masked = per_token * valid
    return jnp.sum(masked) / jnp.maximum(jnp.sum(valid), 1)


def _vocab_parallel_per_token_ce(
    local_logits: jax.Array,
    local_labels: jax.Array,
    local_vocab_ids: jax.Array,
    model_axis: str,
) -> jax.Array:
    """Per-token cross-entropy for one vocab shard, run inside a ``shard_map``.

    Each device only owns a contiguous slice of the vocabulary, so this body
    runs once per ``model_axis`` shard on *local* tensors and combines the
    shards with explicit collectives. We only ever use ``psum`` (sum
    all-reduce) and ``all_gather``, because the TT backend cannot legalize a
    max all-reduce (``pmax``) nor ``partition_id`` (``jax.lax.axis_index``).

    Args:
        local_logits: This shard's logits slice, shape ``[b, s, v_local]``.
        local_labels: The full (replicated) labels, shape ``[b, s]``. Same on
            every shard since the label dim is not sharded.
        local_vocab_ids: The *global* vocabulary ids this shard owns, shape
            ``[v_local]``. This replaces ``axis_index``-based offset maths:
            position ``j`` in this shard corresponds to global token id
            ``local_vocab_ids[j]``.
        model_axis: Name of the mesh axis the vocab dim is sharded over.

    Returns:
        Per-token loss, shape ``[b, s-1]`` (causal shift drops the last pos),
        identical on every ``model_axis`` shard.
    """
    # Causal shift: predict token t+1 from the logits at position t.
    shift_logits = local_logits[:, :-1, :].astype(jnp.float32)  # [b, s-1, v_local]
    shift_labels = local_labels[:, 1:].astype(jnp.int32)  # [b, s-1]

    # --- Stable global log-sum-exp over the full (sharded) vocabulary --------
    # CE = log(sum_v exp(logit_v)) - logit_target. We subtract the global max
    # before exp so no shard overflows. The global max is obtained by
    # all-gathering each shard's (tiny [b, s-1, 1]) local max and reducing
    # locally, since a max all-reduce is unsupported on TT.
    local_max = jnp.max(shift_logits, axis=-1, keepdims=True)  # [b, s-1, 1]
    gathered_max = jax.lax.all_gather(local_max, model_axis, axis=-1, tiled=True)  # [b, s-1, n_model]
    global_max = jnp.max(gathered_max, axis=-1, keepdims=True)  # [b, s-1, 1]

    shifted = shift_logits - global_max  # all <= 0, so exp() can't overflow
    local_sum_exp = jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True)  # [b, s-1, 1]
    global_sum_exp = jax.lax.psum(local_sum_exp, model_axis)  # sum exps across shards
    log_sum_exp = jnp.log(global_sum_exp) + global_max  # [b, s-1, 1] (undo the shift)

    # --- Logit of the target token ------------------------------------------
    # Compare each label to the global ids this shard owns. The owning shard
    # has exactly one True per token; all other shards are all-False and
    # contribute 0, so the psum picks out the correct target logit. A plain
    # equality compare avoids both a gather and any offset arithmetic.
    match = shift_labels[..., None] == local_vocab_ids[None, None, :]  # [b, s-1, v_local]
    local_target = jnp.sum(jnp.where(match, shift_logits, 0.0), axis=-1)  # [b, s-1]
    target_logit = jax.lax.psum(local_target, model_axis)  # [b, s-1]

    return log_sum_exp[..., 0] - target_logit  # [b, s-1]


def _vocab_parallel_loss_scalar(
    local_logits: jax.Array,
    local_labels: jax.Array,
    local_vocab_ids: jax.Array,
    model_axis: str,
    data_axis: str | None,
    ignored_index: int,
) -> jax.Array:
    per_token = _vocab_parallel_per_token_ce(
        local_logits, local_labels, local_vocab_ids, model_axis
    )  # [b, s-1]

    shift_labels = local_labels[:, 1:].astype(jnp.int32)
    valid = shift_labels != ignored_index

    local_loss_sum = jnp.sum(per_token * valid.astype(per_token.dtype))
    local_valid_count = jnp.sum(valid.astype(jnp.int32))

    # IMPORTANT: reduce across data axis only.
    # Do NOT psum across model axis here, because per_token is already identical
    # on all model shards; psum over model would overcount.
    if data_axis is not None:
        local_loss_sum = jax.lax.psum(local_loss_sum, data_axis)
        local_valid_count = jax.lax.psum(local_valid_count, data_axis)

    return local_loss_sum / jnp.maximum(local_valid_count.astype(local_loss_sum.dtype), 1.0)


def vocab_parallel_cross_entropy(
    logits: jax.Array,
    labels: jax.Array,
    mesh: Mesh,
    *,
    model_axis: str = "model",
    data_axis: str = "data",
    ignored_index: int = IGNORED_LABEL,
) -> jax.Array:
    data_spec = data_axis if data_axis in mesh.axis_names else None
    vocab_size = logits.shape[-1]
    vocab_ids = jnp.arange(vocab_size, dtype=jnp.int32)

    return shard_map(
        lambda lg, lb, vids: _vocab_parallel_loss_scalar(
            lg, lb, vids, model_axis, data_spec, ignored_index
        ),
        mesh=mesh,
        in_specs=(
            PartitionSpec(data_spec, None, model_axis),
            PartitionSpec(data_spec, None),
            PartitionSpec(model_axis),
        ),
        out_specs=PartitionSpec(),  # scalar
        check_rep=False,
    )(logits, labels, vocab_ids)


def _vocab_parallel_embedding_lookup_local(
    local_embedding: jax.Array,
    local_input_ids: jax.Array,
    local_vocab_ids: jax.Array,
    model_axis: str,
) -> jax.Array:
    """Embed one vocab shard's tokens, run inside a ``shard_map``.

    The tied embedding table is split along the vocab (row) axis across
    ``model_axis``, so EasyDel's ``jnp.take`` row-gather is not legalizable on
    TT. Instead each shard builds a one-hot over the vocab ids it owns and
    matmuls it with its rows; only the shard that owns a token contributes a
    non-zero row, so a ``psum`` over ``model_axis`` reconstructs the full
    embedding. Uses only matmul + ``psum`` (no gather, no ``axis_index``).

    Args:
        local_embedding: This shard's embedding rows, shape ``[v_local, hidden]``.
        local_input_ids: Token ids, shape ``[b, s]`` (replicated over model).
        local_vocab_ids: Global vocab ids this shard owns, shape ``[v_local]``.
        model_axis: Mesh axis the vocab dim is sharded over.

    Returns:
        Embedded tokens, shape ``[b, s, hidden]``, identical on every
        ``model_axis`` shard.
    """
    # [b, s, v_local]: True only where a token id belongs to this shard's slice.
    match = local_input_ids[..., None] == local_vocab_ids[None, None, :]
    one_hot = match.astype(local_embedding.dtype)

    # [b, s, v_local] @ [v_local, hidden] -> [b, s, hidden]. Shards that do not
    # own a token contribute all-zero rows, so the cross-shard sum selects the
    # correct embedding row. A one-hot matmul avoids the row gather (jnp.take).
    local_embeds = jnp.matmul(one_hot, local_embedding)
    return jax.lax.psum(local_embeds, model_axis)


def vocab_parallel_embedding_lookup(
    embedding: jax.Array,
    input_ids: jax.Array,
    mesh: Mesh,
    *,
    model_axis: str = "model",
    data_axis: str = "data",
) -> jax.Array:
    """Input embedding lookup for a vocab(row)-sharded tied embedding table.

    Mirrors ``vocab_parallel_cross_entropy``: the per-shard work runs inside a
    ``shard_map`` and the shards are combined with ``psum`` over ``model_axis``.
    This replaces EasyDel's ``Embed.__call__`` (``jnp.take`` over axis 0), which
    fails on TT when the embedding rows are sharded across ``model_axis``.

    The result is fed to the model as ``inputs_embeds`` so the model never runs
    its own row-sharded gather.

    Args:
        embedding: Tied embedding table, shape ``[vocab, hidden]``, sharded
            ``[model, None]`` (vocab on ``model_axis``).
        input_ids: Token ids, shape ``[batch, seq]``.
        mesh: Device mesh; must contain ``model_axis``.
        model_axis: Mesh axis the vocab dim is sharded over.
        data_axis: Mesh axis the batch dim is sharded over (None-spec if absent).

    Returns:
        inputs_embeds, shape ``[batch, seq, hidden]`` (hidden replicated; the
        model reshards it to its expected hidden-state sharding).
    """
    vocab_size = embedding.shape[0]
    # Batch dim carries a real axis name only if the mesh has it; else replicated.
    data_spec = data_axis if data_axis in mesh.axis_names else None

    # Global vocab ids 0..V-1 sharded exactly like the embedding's vocab/row dim,
    # so each shard receives the ids of the rows it owns (no axis_index needed).
    vocab_ids = jnp.arange(vocab_size, dtype=jnp.int32)

    # in_specs / out_specs describe how each global array is split:
    #   embedding -> vocab(rows) on model, hidden replicated
    #   input_ids -> batch on data, replicated over model
    #   vocab_ids -> sharded on model (matches the embedding's row split)
    #   embeds    -> batch on data, seq + hidden replicated (psum makes shards agree)
    return shard_map(
        lambda emb, ids, vids: _vocab_parallel_embedding_lookup_local(
            emb, ids.astype(jnp.int32), vids, model_axis
        ),
        mesh=mesh,
        in_specs=(
            PartitionSpec(model_axis, None),
            PartitionSpec(data_spec, None),
            PartitionSpec(model_axis),
        ),
        out_specs=PartitionSpec(data_spec, None, None),
        check_rep=False,
    )(embedding, input_ids, vocab_ids)


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
