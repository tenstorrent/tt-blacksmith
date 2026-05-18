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


def show_predictions(
    collected: list[dict],
    tokenizer,
    *,
    num_tokens: int = 20,
    ignored_index: int = IGNORED_LABEL,
    training_logger: Optional[TrainingLogger] = None,
) -> None:
    """Print collected prediction examples (CPU-only, no forward pass).

    Args:
        collected: List of dicts with keys input_ids, labels, predictions,
            per_token_loss (numpy arrays).
        tokenizer: HuggingFace tokenizer for decoding.
        num_tokens: Number of leading tokens to show per example.
        ignored_index: Label value treated as "don't care".
        training_logger: Optional TrainingLogger; falls back to the
            module-level logger when None.
    """
    log = training_logger.info if training_logger is not None else logging.getLogger(__name__).info

    for i, ex in enumerate(collected):
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        predictions = ex["predictions"]
        per_token_loss = ex["per_token_loss"]

        shift_labels = labels[1:].astype(np.int32)
        target_ids = shift_labels[:num_tokens]
        pred_ids = predictions[:num_tokens]
        token_losses = per_token_loss[:num_tokens]

        tok_valid = target_ids != ignored_index
        valid_targets = target_ids[tok_valid]
        valid_preds = pred_ids[tok_valid]

        input_text = tokenizer.decode(
            input_ids.tolist(),
            skip_special_tokens=True,
        )[:200]
        target_text = tokenizer.decode(
            valid_targets.tolist(),
            skip_special_tokens=False,
        )
        pred_text = tokenizer.decode(
            valid_preds.tolist(),
            skip_special_tokens=False,
        )

        valid = shift_labels != ignored_index
        correct = int((predictions[valid] == shift_labels[valid]).sum())
        total = int(valid.sum())

        log(f"\n--- Example {i + 1} ---")
        log(f"  Input:        {input_text!r}")
        log(f"  Target IDs:   {target_ids.tolist()}")
        log(f"  Pred IDs:     {pred_ids.tolist()}")
        log(f"  Target text:  {target_text!r}")
        log(f"  Pred text:    {pred_text!r}")
        log(f"  Token losses: {np.round(token_losses, 4).tolist()}")
        mean_loss = float(per_token_loss.mean())
        log(f"  Mean loss:    {mean_loss:.4f}")
        acc = correct / max(total, 1)
        log(f"  Accuracy:     {correct}/{total} = {acc:.3f}")
