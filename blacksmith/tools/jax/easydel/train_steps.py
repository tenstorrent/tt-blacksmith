# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec
from transformers import PreTrainedTokenizerBase

from blacksmith.tools.jax.device_manager import JaxDeviceManager
from blacksmith.tools.jax.helpers import (
    masked_cross_entropy,
    show_predictions,
    vocab_parallel_cross_entropy,
)
from blacksmith.tools.logging_manager import TrainingLogger


def _make_loss_fn(mesh, model_axis: str = "model") -> Callable:
    """Pick the cross-entropy matching the active sharding.

    Vocab-parallel CE when the vocab axis is sharded (model_axis size > 1) so
    the softmax normaliser is reduced across shards; otherwise the clamped CE.
    """
    vocab_parallel = mesh is not None and model_axis in getattr(mesh, "axis_names", ()) and mesh.shape[model_axis] > 1

    def cross_entropy(logits, labels):
        if vocab_parallel:
            return vocab_parallel_cross_entropy(logits, labels, mesh, model_axis=model_axis)
        return masked_cross_entropy(logits, labels, clamped=True)

    return cross_entropy


def _make_forward_fn(mesh, model_axis: str = "model") -> Callable:
    """Return forward(model, input_ids, attention_mask) -> logits.

    The model embeds input_ids on device.
    """

    def forward(model, input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    return forward


def create_fused_train_step_fn(
    graphdef: nnx.GraphDef,
    tx: optax.GradientTransformation,
    lora_shardings,
    mesh=None,
    model_axis: str = "model",
) -> Callable:
    """JIT-compiled forward + backward + optimizer step.

    Shift/mask/one-hot/clamped CE all run inside the JIT so no per-step
    micro-ops escape to TT and trigger fabric/MeshDevice re-init.

    Returns fused_train_step(lora_params, frozen_state, opt_state,
    input_ids, labels, attention_mask) -> (new_params, new_opt, loss,
    grad_stats).
    """

    cross_entropy = _make_loss_fn(mesh, model_axis)
    forward = _make_forward_fn(mesh, model_axis)
    scalar_sharding = NamedSharding(mesh, PartitionSpec()) if mesh is not None else None

    def _pin_scalar(x):
        if scalar_sharding is None:
            return x
        return jax.lax.with_sharding_constraint(x, scalar_sharding)

    def loss_fn(lora_params, frozen_state, input_ids, labels, attention_mask):
        model = nnx.merge(graphdef, lora_params, frozen_state)
        logits = forward(model, input_ids, attention_mask)
        logits = jax.lax.with_sharding_constraint(logits, NamedSharding(mesh, PartitionSpec(*([None] * logits.ndim))))
        return cross_entropy(logits, labels)

    @jax.jit
    def fused_train_step(
        lora_params,
        frozen_state,
        opt_state,
        input_ids,
        labels,
        attention_mask,
    ):
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(
            lora_params,
            frozen_state,
            input_ids,
            labels,
            attention_mask,
        )

        grads = jax.tree.map(
            lambda grad, sharding: jax.lax.with_sharding_constraint(grad, sharding),
            grads,
            lora_shardings,
        )

        updates, new_opt = tx.update(grads, opt_state, lora_params)
        new_params = optax.apply_updates(lora_params, updates)
        new_params = jax.tree.map(jax.lax.optimization_barrier, new_params)
        # leaves = jax.tree.leaves(grads)
        # grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
        # grad_max = jnp.max(jnp.stack([jnp.max(jnp.abs(g)) for g in leaves]))
        loss = _pin_scalar(loss)
        # grad_norm = _pin_scalar(grad_norm)
        # grad_max = _pin_scalar(grad_max)
        return new_params, new_opt, loss  # {"grad_norm": grad_norm, "grad_max": grad_max}

    return fused_train_step


def create_eval_step_fn(
    graphdef: nnx.GraphDef,
    mesh=None,
    model_axis: str = "model",
) -> Callable:
    """JIT-compiled evaluation step returning a scalar loss.

    Uses the vocab-parallel CE when the vocab dim is sharded, else the clamped
    CE. A single fixed-signature JIT avoids the MeshDevice re-init crashes that
    alternating JITs caused on TT. Predictions are not emitted; see eval_step.

    Returns eval_step(lora_params, frozen_state, input_ids, labels,
    attention_mask) -> scalar loss.
    """

    cross_entropy = _make_loss_fn(mesh, model_axis)
    forward = _make_forward_fn(mesh, model_axis)

    def eval_step(lora_params, frozen_state, input_ids, labels, attention_mask):
        model = nnx.merge(graphdef, lora_params, frozen_state)
        logits = forward(model, input_ids, attention_mask)
        logits = jax.lax.with_sharding_constraint(logits, NamedSharding(mesh, PartitionSpec(*[None] * logits.ndim)))
        # Loss only: extra outputs (e.g. argmax predictions) break the
        # single fixed-signature eval JIT on TT.
        return cross_entropy(logits, labels)

    # Pin the scalar loss to a replicated 0-D sharding; otherwise TT infers a
    # non-0-D sharding for it and compilation fails.
    # TODO(ndimicTT): remove once TT partitions 0-D scalar outputs correctly.
    if mesh is not None:
        return jax.jit(eval_step, out_shardings=NamedSharding(mesh, PartitionSpec()))
    return jax.jit(eval_step)


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
    """Run evaluation over validation batches and return the average loss.

    When a tokenizer is provided, the first num_examples batches are also moved
    to CPU and shown via show_predictions (input/target/loss only).
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
