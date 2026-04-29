# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

IGNORED_LABEL = -100

# Epsilon used inside log to keep gradients finite when a row of the
# renormalized softmax has near-zero probability.
_LOG_EPS = 1e-12


def _clamped_softmax_cross_entropy_per_token(logits_f32, one_hot):
    """Per-token cross-entropy that is robust to TT bf16 fused-softmax drift.

    On TT the fused softmax kernel in bf16 can produce rows that do not
    sum to 1 and occasionally individual entries > 1.0, which makes
    optax.softmax_cross_entropy yield slightly negative per-token values.

    All on-device:
      1. compute softmax,
      2. clamp to [0, 1],
      3. renormalize so each row sums to 1,
      4. take the standard -sum(one_hot * log(probs)).
    """
    probs = jax.nn.softmax(logits_f32, axis=-1)
    probs = jnp.clip(probs, 0.0, 1.0)
    probs = probs / jnp.maximum(jnp.sum(probs, axis=-1, keepdims=True), _LOG_EPS)
    log_probs = jnp.log(jnp.maximum(probs, _LOG_EPS))
    return -jnp.sum(one_hot * log_probs, axis=-1)


def create_train_step_fn(
    graphdef: nnx.GraphDef,
    optimizer: optax.GradientTransformation,
) -> Callable:
    """Create a JIT-compiled training step (fwd + bwd + optimizer).

    One-hot labels and label_mask are pre-computed outside JIT.
    On TT this avoids a ttnn.eq bug that doubles the one-hot
    value for even uint32 labels.

    Signature of the returned function:

        train_step(lora_params, frozen_state, opt_state,
                   input_ids, one_hot_labels, label_mask,
                   attention_mask)
            -> (loss, new_lora_params, new_opt_state, grad_stats)
    """

    def loss_fn(
        lora_params,
        frozen_state,
        input_ids,
        one_hot_labels,
        label_mask,
        attention_mask,
    ):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        out = m(input_ids=input_ids, attention_mask=attention_mask)

        # NOTE (TT): TT-MLIR may run softmax in bf16 inside fused graphs,
        # which drifts the row sum away from 1 and can push individual
        # entries above 1. We compensate with a clamp-and-renormalize CE;
        # see _clamped_softmax_cross_entropy_per_token.
        shift_logits = out.logits[:, :-1, :].astype(jnp.float32)
        per_token = _clamped_softmax_cross_entropy_per_token(
            shift_logits,
            one_hot_labels,
        )
        masked = per_token * label_mask
        return jnp.sum(masked) / jnp.maximum(
            jnp.sum(label_mask),
            1.0,
        )

    def train_step(
        lora_params,
        frozen_state,
        opt_state,
        input_ids,
        one_hot_labels,
        label_mask,
        attention_mask,
    ):
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
            lora_params,
            frozen_state,
            input_ids,
            one_hot_labels,
            label_mask,
            attention_mask,
        )
        leaves = jax.tree.leaves(grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
        grad_max = jnp.max(
            jnp.stack([jnp.max(jnp.abs(g)) for g in leaves]),
        )
        updates, new_opt = optimizer.update(grads, opt_state, lora_params)
        new_lora = optax.apply_updates(lora_params, updates)
        stats = {"grad_norm": grad_norm, "grad_max": grad_max}
        return loss, new_lora, new_opt, stats

    return jax.jit(train_step)


def _clamped_cross_entropy(logits, labels, ignored_index=IGNORED_LABEL):
    """On-device CE with clamp+renorm to defeat TT bf16 softmax drift.

    Positions where labels == ignored_index are excluded from the mean.
    """
    shift_logits = logits[:, :-1, :].astype(jnp.float32)
    shift_labels = labels[:, 1:].astype(jnp.int32)

    valid = shift_labels != ignored_index
    safe = jnp.where(valid, shift_labels, 0)
    one_hot = jax.nn.one_hot(
        safe,
        shift_logits.shape[-1],
    ).astype(jnp.float32)
    per_token = _clamped_softmax_cross_entropy_per_token(shift_logits, one_hot)
    masked = per_token * valid
    return jnp.sum(masked) / jnp.maximum(jnp.sum(valid), 1)


def _clamped_cross_entropy_with_predictions(
    logits,
    labels,
    ignored_index=IGNORED_LABEL,
):
    """Like _clamped_cross_entropy but also returns predictions and per-token losses."""
    shift_logits = logits[:, :-1, :].astype(jnp.float32)
    shift_labels = labels[:, 1:].astype(jnp.int32)

    valid = shift_labels != ignored_index
    safe = jnp.where(valid, shift_labels, 0)
    one_hot = jax.nn.one_hot(
        safe,
        shift_logits.shape[-1],
    ).astype(jnp.float32)
    per_token = _clamped_softmax_cross_entropy_per_token(shift_logits, one_hot)
    masked = per_token * valid
    loss = jnp.sum(masked) / jnp.maximum(jnp.sum(valid), 1)
    predictions = jnp.argmax(shift_logits, axis=-1)
    return loss, predictions, per_token


def create_eval_step_fn(graphdef: nnx.GraphDef) -> Callable:
    """Create an evaluation step.

    Fully on-device for all devices. On TT the softmax-in-CE is guarded by
    _clamped_softmax_cross_entropy_per_token (clamp to [0, 1] + renormalize)
    to correct the bf16 fused-softmax drift observed on TT.

    Signature:

        eval_step(lora_params, frozen_state,
                  input_ids, labels, attention_mask) -> loss

    Labels may contain -100 at masked positions.
    """

    @jax.jit
    def eval_step(
        lora_params,
        frozen_state,
        input_ids,
        labels,
        attention_mask,
    ):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        logits = m(input_ids=input_ids, attention_mask=attention_mask).logits
        return _clamped_cross_entropy(logits, labels)

    return eval_step


def create_eval_inspect_step_fn(graphdef: nnx.GraphDef) -> Callable:
    """Eval step returning (loss, predictions, per_token_loss).

    Fully on-device; on TT the CE uses clamp+renorm via
    _clamped_cross_entropy_with_predictions to correct bf16 softmax drift.
    """

    @jax.jit
    def eval_inspect_step(
        lora_params,
        frozen_state,
        input_ids,
        labels,
        attention_mask,
    ):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        logits = m(input_ids=input_ids, attention_mask=attention_mask).logits
        return _clamped_cross_entropy_with_predictions(logits, labels)

    return eval_inspect_step


def _show_predictions(collected, tokenizer, num_tokens=20, max_input_chars=200):
    """Print collected prediction examples (CPU-only, no forward pass).

    Args:
        collected: List of dicts with keys input_ids, labels,
            predictions, per_token_loss (numpy arrays).
        tokenizer: HuggingFace tokenizer for decoding.
        num_tokens: Leading tokens to show per example.
        max_input_chars: Truncate the printed prompt to this many chars
            so a long input does not flood the log.
    """
    for i, ex in enumerate(collected):
        input_ids = ex["input_ids"]
        labels = ex["labels"]
        predictions = ex["predictions"]
        per_token_loss = ex["per_token_loss"]

        shift_labels = labels[1:].astype(np.int32)
        target_ids = shift_labels[:num_tokens]
        pred_ids = predictions[:num_tokens]
        token_losses = per_token_loss[:num_tokens]

        # Filter out IGNORED_LABEL (-100) before decoding: the
        # tokenizer expects unsigned IDs and overflows on negatives.
        tok_valid = target_ids != IGNORED_LABEL
        valid_targets = target_ids[tok_valid]
        valid_preds = pred_ids[tok_valid]

        input_text = tokenizer.decode(
            input_ids.tolist(),
            skip_special_tokens=True,
        )[:max_input_chars]
        target_text = tokenizer.decode(
            valid_targets.tolist(),
            skip_special_tokens=False,
        )
        pred_text = tokenizer.decode(
            valid_preds.tolist(),
            skip_special_tokens=False,
        )

        valid = shift_labels != IGNORED_LABEL
        correct = int((predictions[valid] == shift_labels[valid]).sum())
        total = int(valid.sum())

        logger.info(f"\n--- Example {i + 1} ---")
        logger.info(f"  Input:        {input_text!r}")
        logger.info(f"  Target IDs:   {target_ids.tolist()}")
        logger.info(f"  Pred IDs:     {pred_ids.tolist()}")
        logger.info(f"  Target text:  {target_text!r}")
        logger.info(f"  Pred text:    {pred_text!r}")
        logger.info(f"  Token losses: " f"{np.round(token_losses, 4).tolist()}")
        logger.info(f"  Mean loss:    {float(per_token_loss.mean()):.4f}")
        logger.info(f"  Accuracy:     {correct}/{total} " f"= {correct / max(total, 1):.3f}")


def evaluate(
    jit_eval_step: Callable,
    lora_params: nnx.State,
    frozen_state: nnx.State,
    val_batches: list[dict[str, np.ndarray]],
    *,
    jit_inspect_step: Optional[Callable] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    num_examples: int = 3,
    num_tokens: int = 20,
) -> float:
    """Run evaluation on validation batches and return average loss.

    Each element of val_batches is a dict with keys
    input_ids, labels, and attention_mask.

    When jit_inspect_step and tokenizer are provided, the first
    few batches also collect decoded prediction examples.
    """
    total_loss = 0.0
    collected: list[dict[str, np.ndarray]] = []
    can_inspect = jit_inspect_step is not None and tokenizer is not None

    for batch in val_batches:
        ids = batch["input_ids"]
        labels = batch["labels"]
        attn = batch["attention_mask"]

        if can_inspect and len(collected) < num_examples:
            loss, preds, ptl = jit_inspect_step(
                lora_params,
                frozen_state,
                ids,
                labels,
                attn,
            )
            cpu = jax.devices("cpu")[0]
            b_ids = np.array(jax.device_put(ids, cpu))
            b_lbl = np.array(jax.device_put(labels, cpu))
            b_preds = np.array(jax.device_put(preds, cpu))
            b_ptl = np.array(jax.device_put(ptl, cpu))
            bs = b_ids.shape[0]
            for idx in range(min(bs, num_examples - len(collected))):
                collected.append(
                    {
                        "input_ids": b_ids[idx],
                        "labels": b_lbl[idx],
                        "predictions": b_preds[idx],
                        "per_token_loss": b_ptl[idx],
                    }
                )
        else:
            loss = jit_eval_step(
                lora_params,
                frozen_state,
                ids,
                labels,
                attn,
            )
        total_loss += float(loss)

    if collected:
        _show_predictions(collected, tokenizer, num_tokens)

    n = len(val_batches)
    return total_loss / n if n else 0.0
