# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from transformers import PreTrainedTokenizerBase

from blacksmith.tools.jax.device_manager import JaxDeviceManager
from blacksmith.tools.jax.helpers import (
    masked_cross_entropy,
    show_predictions,
)
from blacksmith.tools.logging_manager import TrainingLogger


def create_fused_train_step_fn(
    graphdef: nnx.GraphDef,
    tx: optax.GradientTransformation,
) -> Callable:
    """JIT-compiled forward + backward + optimizer step.

    Label shift, masking, one-hot, and clamped CE run inside the JIT via
    masked_cross_entropy so no per-step micro-ops escape to TT and
    trigger fabric / MeshDevice re-init (tt-xla#1993, tt-xla#4809).

    Returns fused_train_step(lora_params, frozen_state, opt_state,
    input_ids, labels, attention_mask) -> (new_params, new_opt, loss,
    grad_stats).
    """

    def loss_fn(lora_params, frozen_state, input_ids, labels, attention_mask):
        model = nnx.merge(graphdef, lora_params, frozen_state)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        return masked_cross_entropy(logits, labels, clamped=True)

    @jax.jit
    def fused_train_step(
        lora_params, frozen_state, opt_state,
        input_ids, labels, attention_mask,
    ):
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
            lora_params, frozen_state,
            input_ids, labels, attention_mask,
        )
        updates, new_opt = tx.update(grads, opt_state, lora_params)
        new_params = optax.apply_updates(lora_params, updates)
        leaves = jax.tree.leaves(grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
        grad_max = jnp.max(jnp.stack([jnp.max(jnp.abs(g)) for g in leaves]))
        return new_params, new_opt, loss, {"grad_norm": grad_norm, "grad_max": grad_max}

    return fused_train_step


def create_eval_step_fn(graphdef: nnx.GraphDef) -> Callable:
    """JIT-compiled evaluation step returning scalar loss.

    Uses the TT-safe clamped CE via masked_cross_entropy(clamped=True).

    Returns eval_step(lora_params, frozen_state, input_ids, labels,
    attention_mask) -> loss.
    """

    @jax.jit
    def eval_step(lora_params, frozen_state, input_ids, labels, attention_mask):
        model = nnx.merge(graphdef, lora_params, frozen_state)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        return masked_cross_entropy(logits, labels, clamped=True)

    return eval_step


def evaluate(
    jit_eval_step: Callable,
    lora_params: nnx.State,
    frozen_state: nnx.State,
    val_batches: list[dict[str, np.ndarray]],
    *,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    num_examples: int = 3,
    num_tokens: int = 20,
    training_logger: Optional[TrainingLogger] = None,
) -> float:
    """Run evaluation on validation batches and return average loss.

    When a tokenizer is provided, the first num_examples batches also
    transfer input_ids/labels to CPU and display them via
    show_predictions.  Only the scalar-loss jit_eval_step runs on-device;
    introducing a second JIT with a different output signature would
    trigger MeshDevice migration crashes on TT (tt-xla#1993, tt-xla#4809).
    """
    total_loss = 0.0
    collected_examples: list[dict[str, np.ndarray]] = []

    for batch in val_batches:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        loss = jit_eval_step(lora_params, frozen_state, input_ids, labels, attention_mask)

        if tokenizer is not None and len(collected_examples) < num_examples:
            batch_input_ids = np.asarray(JaxDeviceManager.to_cpu(input_ids))
            batch_labels = np.asarray(JaxDeviceManager.to_cpu(labels))
            batch_size = batch_input_ids.shape[0]
            for i in range(min(batch_size, num_examples - len(collected_examples))):
                collected_examples.append(
                    {
                        "input_ids": batch_input_ids[i],
                        "labels": batch_labels[i],
                        "loss": float(loss),
                    }
                )

        total_loss += float(loss)

    if collected_examples:
        show_predictions(
            collected_examples,
            tokenizer,
            num_tokens=num_tokens,
            training_logger=training_logger,
        )

    num_batches = len(val_batches)
    return total_loss / num_batches if num_batches else 0.0
