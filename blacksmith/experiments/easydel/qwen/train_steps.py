# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""JIT-compiled training / evaluation steps and evaluation helpers."""

import inspect
import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

logger = logging.getLogger(__name__)

IGNORED_LABEL = -100


def _build_model_kwargs(call_signature, input_ids, attention_mask, *, train):
    """Build model ``__call__`` kwargs, adding only supported args."""
    kwargs = {"input_ids": input_ids}
    if "attention_mask" in call_signature.parameters:
        kwargs["attention_mask"] = attention_mask
    if "train" in call_signature.parameters:
        kwargs["train"] = train
    if "deterministic" in call_signature.parameters:
        kwargs["deterministic"] = not train
    return kwargs


# ------------------------------------------------------------------
# Training step
# ------------------------------------------------------------------


def create_train_step_fn(
    graphdef: Any,
    call_signature: inspect.Signature,
    tx: Any,
) -> Any:
    """Create a JIT-compiled training step (fwd + bwd + optimizer).

    One-hot labels and ``label_mask`` are pre-computed outside JIT.
    On TT this avoids a ``ttnn.eq`` bug that doubles the one-hot
    value for even uint32 labels.

    Signature of the returned function::

        train_step(lora_params, frozen_state, opt_state,
                   input_ids, one_hot_labels, label_mask,
                   attention_mask, *, train)
            -> (loss, new_lora_params, new_opt_state, grad_stats)

    """

    def loss_fn(
        lora_params,
        frozen_state,
        input_ids,
        one_hot_labels,
        label_mask,
        attention_mask,
        *,
        train,
    ):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        kwargs = _build_model_kwargs(
            call_signature,
            input_ids,
            attention_mask,
            train=train,
        )
        out = m(**kwargs)

        # NOTE (TT): TT-MLIR may run softmax in bf16 inside fused
        # graphs.  Gradient direction is preserved; magnitude is
        # approximate.  Rely on CPU f32 eval for accurate metrics.
        shift_logits = out.logits[:, :-1, :].astype(jnp.float32)
        per_token = optax.softmax_cross_entropy(
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
        *,
        train,
    ):
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
            lora_params,
            frozen_state,
            input_ids,
            one_hot_labels,
            label_mask,
            attention_mask,
            train=train,
        )
        leaves = jax.tree.leaves(grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
        grad_max = jnp.max(
            jnp.stack([jnp.max(jnp.abs(g)) for g in leaves]),
        )
        updates, new_opt = tx.update(grads, opt_state, lora_params)
        new_lora = optax.apply_updates(lora_params, updates)
        stats = {"grad_norm": grad_norm, "grad_max": grad_max}
        return loss, new_lora, new_opt, stats

    return jax.jit(train_step, static_argnames=("train",))


# ------------------------------------------------------------------
# Forward / eval helpers
# ------------------------------------------------------------------


def _create_forward_fn(
    graphdef: Any,
    call_signature: inspect.Signature,
) -> Any:
    """JIT-compiled forward returning raw logits (no loss).

    Used by TT evaluation path: logits are computed on device, then
    transferred to CPU for f32 loss computation.
    """

    @jax.jit
    def forward_fn(lora_params, frozen_state, input_ids, attention_mask):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        kwargs = _build_model_kwargs(
            call_signature,
            input_ids,
            attention_mask,
            train=False,
        )
        return m(**kwargs).logits

    return forward_fn


def _cpu_cross_entropy(logits, labels, ignored_index=IGNORED_LABEL):
    """CE loss on CPU host in f32, with label masking.

    Transfers logits to CPU to avoid TT bf16 softmax inflation.
    Positions where ``labels == ignored_index`` are excluded from
    the mean.
    """
    cpu = jax.devices("cpu")[0]
    logits_f32 = jax.device_put(logits, cpu).astype(jnp.float32)
    labels_cpu = jax.device_put(labels, cpu)

    shift_logits = logits_f32[:, :-1, :]
    shift_labels = labels_cpu[:, 1:].astype(jnp.int32)

    valid = shift_labels != ignored_index
    safe = jnp.where(valid, shift_labels, 0)
    one_hot = jax.nn.one_hot(
        safe,
        shift_logits.shape[-1],
    ).astype(jnp.float32)
    per_token = optax.softmax_cross_entropy(shift_logits, one_hot)
    masked = per_token * valid
    return jnp.sum(masked) / jnp.maximum(jnp.sum(valid), 1)


def _cpu_cross_entropy_with_inspect(
    logits,
    labels,
    ignored_index=IGNORED_LABEL,
):
    """Like ``_cpu_cross_entropy`` but also returns predictions
    and per-token losses."""
    cpu = jax.devices("cpu")[0]
    logits_f32 = jax.device_put(logits, cpu).astype(jnp.float32)
    labels_cpu = jax.device_put(labels, cpu)

    shift_logits = logits_f32[:, :-1, :]
    shift_labels = labels_cpu[:, 1:].astype(jnp.int32)

    valid = shift_labels != ignored_index
    safe = jnp.where(valid, shift_labels, 0)
    one_hot = jax.nn.one_hot(
        safe,
        shift_logits.shape[-1],
    ).astype(jnp.float32)
    per_token = optax.softmax_cross_entropy(shift_logits, one_hot)
    masked = per_token * valid
    loss = jnp.sum(masked) / jnp.maximum(jnp.sum(valid), 1)
    predictions = jnp.argmax(shift_logits, axis=-1)
    return loss, predictions, per_token


# ------------------------------------------------------------------
# Evaluation step builders
# ------------------------------------------------------------------


def create_eval_step_fn(
    graphdef: Any,
    call_signature: inspect.Signature,
    *,
    device_kind: str = "tt",
) -> Any:
    """Create an evaluation step.

    On TT: forward on device, loss on CPU (bf16 workaround).
    On GPU/CPU: fully on-device eval in f32.

    Signature::

        eval_step(lora_params, frozen_state,
                  input_ids, labels, attention_mask) -> loss

    Labels may contain ``-100`` at masked positions.
    """
    if device_kind == "tt":
        jit_fwd = _create_forward_fn(graphdef, call_signature)

        def eval_step(
            lora_params,
            frozen_state,
            input_ids,
            labels,
            attention_mask,
        ):
            logits = jit_fwd(
                lora_params,
                frozen_state,
                input_ids,
                attention_mask,
            )
            return _cpu_cross_entropy(logits, labels)

        return eval_step

    @jax.jit
    def eval_step(
        lora_params,
        frozen_state,
        input_ids,
        labels,
        attention_mask,
    ):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        kwargs = _build_model_kwargs(
            call_signature,
            input_ids,
            attention_mask,
            train=False,
        )
        logits = m(**kwargs).logits
        shift_logits = logits[:, :-1, :].astype(jnp.float32)
        shift_labels = labels[:, 1:].astype(jnp.int32)

        valid = shift_labels != IGNORED_LABEL
        safe = jnp.where(valid, shift_labels, 0)
        one_hot = jax.nn.one_hot(safe, shift_logits.shape[-1])
        per_token = optax.softmax_cross_entropy(shift_logits, one_hot)
        masked = per_token * valid
        return jnp.sum(masked) / jnp.maximum(jnp.sum(valid), 1)

    return eval_step


def create_eval_inspect_step_fn(
    graphdef: Any,
    call_signature: inspect.Signature,
    *,
    device_kind: str = "tt",
) -> Any:
    """Eval step returning ``(loss, predictions, per_token_loss)``.

    Same device dispatch as :func:`create_eval_step_fn`.
    """
    if device_kind == "tt":
        jit_fwd = _create_forward_fn(graphdef, call_signature)

        def eval_inspect_step(
            lora_params,
            frozen_state,
            input_ids,
            labels,
            attention_mask,
        ):
            logits = jit_fwd(
                lora_params,
                frozen_state,
                input_ids,
                attention_mask,
            )
            return _cpu_cross_entropy_with_inspect(logits, labels)

        return eval_inspect_step

    @jax.jit
    def eval_inspect_step(
        lora_params,
        frozen_state,
        input_ids,
        labels,
        attention_mask,
    ):
        m = nnx.merge(graphdef, lora_params, frozen_state)
        kwargs = _build_model_kwargs(
            call_signature,
            input_ids,
            attention_mask,
            train=False,
        )
        logits = m(**kwargs).logits
        shift_logits = logits[:, :-1, :].astype(jnp.float32)
        shift_labels = labels[:, 1:].astype(jnp.int32)

        valid = shift_labels != IGNORED_LABEL
        safe = jnp.where(valid, shift_labels, 0)
        one_hot = jax.nn.one_hot(safe, shift_logits.shape[-1])
        per_token = optax.softmax_cross_entropy(shift_logits, one_hot)
        masked = per_token * valid
        loss = jnp.sum(masked) / jnp.maximum(jnp.sum(valid), 1)
        predictions = jnp.argmax(shift_logits, axis=-1)
        return loss, predictions, per_token

    return eval_inspect_step


# ------------------------------------------------------------------
# Evaluation loop + prediction display
# ------------------------------------------------------------------


def _show_predictions(collected, tokenizer, num_tokens=20):
    """Print collected prediction examples (CPU-only, no forward pass).

    Args:
        collected: List of dicts with keys ``input_ids``, ``labels``,
            ``predictions``, ``per_token_loss`` (numpy arrays).
        tokenizer: HuggingFace tokenizer for decoding.
        num_tokens: Leading tokens to show per example.

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
        )[:200]
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
    jit_eval_step: Any,
    lora_params: Any,
    frozen_state: Any,
    val_batches: list[dict[str, Any]],
    *,
    jit_inspect_step: Any = None,
    tokenizer: Any = None,
    num_examples: int = 3,
    num_tokens: int = 20,
) -> float:
    """Run evaluation on validation batches and return average loss.

    Each element of *val_batches* is a dict with keys
    ``input_ids``, ``labels``, and ``attention_mask``.

    When *jit_inspect_step* and *tokenizer* are provided, the first
    few batches also collect decoded prediction examples.
    """
    total_loss = 0.0
    collected: list[dict[str, Any]] = []
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
