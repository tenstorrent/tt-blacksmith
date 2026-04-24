# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


# Optimizer schedule with linear warmup and linear decay.
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


# KL Divergence between two sets of logits with temperature scaling.
def kl_divergence(p_logits, q_logits, T):
    p = nn.softmax(p_logits / T, axis=-1)
    log_p = jax.nn.log_softmax(p_logits / T, axis=-1)
    log_q = jax.nn.log_softmax(q_logits / T, axis=-1)
    kl = jnp.sum(p * (log_p - log_q), axis=-1)
    return (T**2) * jnp.mean(kl)


# Cross-entropy loss with integer labels.
def ce_with_labels(logits, labels):
    num_classes = logits.shape[-1]
    one_hot_labels = jax.nn.one_hot(labels, num_classes)
    return optax.softmax_cross_entropy(logits, one_hot_labels).mean()


# Cosine embedding loss between two sets of vectors.
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
    """Per-token cross-entropy that defeats TT bf16 fused-softmax drift.

    On TT the fused ``softmax`` kernel in bf16 can produce rows that
    do not sum to 1 (observed ~+/-2% drift) and occasionally
    individual entries > 1.0.  Plain
    :func:`optax.softmax_cross_entropy` then yields a per-token term
    that can go slightly negative.

    Fix (all on-device):
      1. compute softmax,
      2. clamp to ``[0, 1]``,
      3. renormalize so each row sums to 1,
      4. ``-sum(one_hot * log(probs))``.
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

    Positions where ``labels == ignored_index`` are excluded
    from the mean.  When *clamped* is ``True`` (default) the
    TT-safe :func:`clamped_softmax_cross_entropy_per_token`
    variant is used; otherwise plain
    :func:`optax.softmax_cross_entropy`.

    Args:
        logits: ``(batch, seq_len, vocab)`` model output.
        labels: ``(batch, seq_len)`` integer labels (may
            contain *ignored_index*).
        ignored_index: Value treated as "don't care".
        clamped: Use the TT bf16-safe CE variant.
        vocab_size: Vocabulary size for one-hot encoding.
            Inferred from *logits* when *None*.
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
