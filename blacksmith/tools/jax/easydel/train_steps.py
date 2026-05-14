# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import PreTrainedTokenizerBase

from blacksmith.tools.jax.device_manager import JaxDeviceManager
from blacksmith.tools.jax.helpers import (
    clamped_softmax_cross_entropy_per_token,
    masked_cross_entropy,
    show_predictions,
)


def create_loss_and_grad_step_fn(graphdef: nnx.GraphDef) -> Callable:
    """Create a JIT-compiled forward + backward step (no optimizer).

    One-hot labels and label_mask are pre-computed outside JIT.
    On TT this avoids a ttnn.eq bug that doubles the one-hot value for
    even uint32 labels.

    Returns a function with signature::

        loss_and_grad_step(lora_params, frozen_state,
                           input_ids, one_hot_labels, label_mask,
                           attention_mask)
            -> (loss, grads, grad_stats)

    The caller is responsible for applying the optimizer update
    (typically via device_manager.optimizer_step).
    """

    def loss_fn(lora_params, frozen_state, input_ids, one_hot_labels, label_mask, attention_mask):
        model = nnx.merge(graphdef, lora_params, frozen_state)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        shift_logits = output.logits[:, :-1, :].astype(jnp.float32)
        per_token_loss = clamped_softmax_cross_entropy_per_token(shift_logits, one_hot_labels)
        masked_per_token_loss = per_token_loss * label_mask
        return jnp.sum(masked_per_token_loss) / jnp.maximum(jnp.sum(label_mask), 1.0)

    @jax.jit
    def loss_and_grad_step(lora_params, frozen_state, input_ids, one_hot_labels, label_mask, attention_mask):
        loss, gradients = jax.value_and_grad(loss_fn, argnums=0)(
            lora_params, frozen_state, input_ids, one_hot_labels, label_mask, attention_mask
        )
        gradient_leaves = jax.tree.leaves(gradients)
        gradient_norm = jnp.sqrt(sum(jnp.sum(gradient_leaf**2) for gradient_leaf in gradient_leaves))
        gradient_max = jnp.max(jnp.stack([jnp.max(jnp.abs(gradient_leaf)) for gradient_leaf in gradient_leaves]))
        return loss, gradients, {"grad_norm": gradient_norm, "grad_max": gradient_max}

    return loss_and_grad_step


def create_eval_step_fn(graphdef: nnx.GraphDef) -> Callable:
    """JIT-compiled evaluation step returning scalar loss.

    Uses the TT-safe clamped CE via masked_cross_entropy(clamped=True)
    so results are correct on all devices.

    Returns a function with signature::

        eval_step(lora_params, frozen_state,
                  input_ids, labels, attention_mask) -> loss
    """

    @jax.jit
    def eval_step(lora_params, frozen_state, input_ids, labels, attention_mask):
        model = nnx.merge(graphdef, lora_params, frozen_state)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        return masked_cross_entropy(logits, labels, clamped=True)

    return eval_step


def create_eval_inspect_step_fn(graphdef: nnx.GraphDef) -> Callable:
    @jax.jit
    def eval_inspect_step(lora_params, frozen_state, input_ids, labels, attention_mask):
        model = nnx.merge(graphdef, lora_params, frozen_state)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].astype(jnp.float32)
        shift_labels = labels[:, 1:].astype(jnp.int32)

        valid_mask = shift_labels != -100
        safe_labels = jnp.where(valid_mask, shift_labels, 0)
        one_hot_labels = jax.nn.one_hot(safe_labels, shift_logits.shape[-1]).astype(jnp.float32)
        per_token_loss = clamped_softmax_cross_entropy_per_token(shift_logits, one_hot_labels)
        masked_per_token_loss = per_token_loss * valid_mask
        loss = jnp.sum(masked_per_token_loss) / jnp.maximum(jnp.sum(valid_mask), 1)
        predictions = jnp.argmax(shift_logits, axis=-1)
        return loss, predictions, per_token_loss

    return eval_inspect_step


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

    When jit_inspect_step and tokenizer are provided, the first few
    batches also collect decoded prediction examples.
    """
    total_loss = 0.0
    collected_examples: list[dict[str, np.ndarray]] = []
    can_inspect = jit_inspect_step is not None and tokenizer is not None

    for batch in val_batches:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        if can_inspect and len(collected_examples) < num_examples:
            loss, predictions, per_token_loss = jit_inspect_step(
                lora_params, frozen_state, input_ids, labels, attention_mask
            )
            batch_input_ids = JaxDeviceManager.to_cpu(input_ids)
            batch_labels = JaxDeviceManager.to_cpu(labels)
            batch_predictions = JaxDeviceManager.to_cpu(predictions)
            batch_per_token_loss = JaxDeviceManager.to_cpu(per_token_loss)
            batch_size = batch_input_ids.shape[0]
            for example_index in range(min(batch_size, num_examples - len(collected_examples))):
                collected_examples.append(
                    {
                        "input_ids": batch_input_ids[example_index],
                        "labels": batch_labels[example_index],
                        "predictions": batch_predictions[example_index],
                        "per_token_loss": batch_per_token_loss[example_index],
                    }
                )
        else:
            loss = jit_eval_step(lora_params, frozen_state, input_ids, labels, attention_mask)
        total_loss += float(loss)

    if collected_examples:
        show_predictions(collected_examples, tokenizer, num_tokens=num_tokens)

    num_batches = len(val_batches)
    return total_loss / num_batches if num_batches else 0.0
