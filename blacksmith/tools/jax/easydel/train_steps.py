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
    embedding_lookup_one_hot_replicated,
    masked_cross_entropy,
    show_predictions,
    vocab_parallel_cross_entropy,
    vocab_parallel_embedding_lookup_gspmd,
)
from blacksmith.tools.logging_manager import TrainingLogger


def _is_vocab_parallel(mesh, model_axis: str) -> bool:
    """True when the model axis exists and has size greater than one.

    With a vocab-sharded output projection the logits are split across
    model_axis, so the softmax normaliser must be reduced across shards with an
    explicit collective, i.e. the vocab-parallel cross-entropy. This is a
    separate decision from how the input embedding is laid out; see
    _make_forward_fn for the embedding side.
    """
    return mesh is not None and model_axis in getattr(mesh, "axis_names", ()) and mesh.shape[model_axis] > 1


def _make_loss_fn(mesh, model_axis: str = "model") -> Callable:
    """Pick the right cross-entropy for the active sharding.

    When the vocab dimension is sharded across ``model_axis`` (size > 1), the
    softmax normaliser must be reduced across shards explicitly, so we use the
    vocab-parallel CE. Otherwise the plain clamped CE is sufficient.
    """
    vocab_parallel = _is_vocab_parallel(mesh, model_axis)

    def cross_entropy(logits, labels):
        if vocab_parallel:
            return vocab_parallel_cross_entropy(logits, labels, mesh, model_axis=model_axis)
        return masked_cross_entropy(logits, labels, clamped=True)

    return cross_entropy


def _make_forward_fn(mesh, model_axis: str = "model", embedding_row_sharded: bool = False) -> Callable:
    """Return forward(model, input_ids, attention_mask) -> logits.

    The embedding layout drives which input path we take, independently of how
    the output logits are sharded. Either way we precompute inputs_embeds with a
    one-hot matmul and feed those to the model, never model(input_ids=...),
    because EasyDel's jnp.take row gather is not legalizable on TT's tile layout:

    - Replicated embedding (embedding_row_sharded is False): replicated one-hot
      matmul (embedding_lookup_one_hot_replicated). Fully replicated, so no
      collectives and no sharded contraction.
    - Row (vocab) sharded embedding (embedding_row_sharded is True): GSPMD one-hot
      matmul lookup (vocab_parallel_embedding_lookup_gspmd) with a contracting-
      dim-sharded matmul.

    Both lookups use GSPMD (no shard_map), so they emit no manual computation
    region. TT accepts at most one such region per module and the vocab-parallel
    cross-entropy already uses one, so the input lookup must not add a second.
    """

    def forward(model, input_ids, attention_mask):
        # [vocab, hidden] embedding table.
        embedding = model.get_embedding().embedding.value
        if embedding_row_sharded:
            # Rows sharded across model_axis: GSPMD one-hot matmul lookup.
            inputs_embeds = vocab_parallel_embedding_lookup_gspmd(
                embedding, input_ids, mesh, model_axis=model_axis
            )
        else:
            # Replicated table: replicated one-hot matmul. We still avoid
            # model(input_ids=...) because EasyDel's jnp.take row gather is not
            # legalizable on TT's tile layout. Feeding inputs_embeds sidesteps it.
            inputs_embeds = embedding_lookup_one_hot_replicated(embedding, input_ids)
        return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

    return forward


def create_fused_train_step_fn(
    graphdef: nnx.GraphDef,
    tx: optax.GradientTransformation,
    lora_shardings,
    mesh=None,
    model_axis: str = "model",
    embedding_row_sharded: bool = False,
) -> Callable:
    """JIT-compiled forward + backward + optimizer step.

    Label shift, masking, one-hot, and clamped CE run inside the JIT via
    masked_cross_entropy so no per-step micro-ops escape to TT and
    trigger fabric / MeshDevice re-init (tt-xla#1993, tt-xla#4809).

    embedding_row_sharded selects the input embedding path: pass True only when
    the token embedding is sharded along its vocab (row) axis, otherwise the
    plain replicated lookup is used. See _make_forward_fn.

    Returns fused_train_step(lora_params, frozen_state, opt_state,
    input_ids, labels, attention_mask) -> (new_params, new_opt, loss,
    grad_stats).
    """

    cross_entropy = _make_loss_fn(mesh, model_axis)
    forward = _make_forward_fn(mesh, model_axis, embedding_row_sharded)

    # Replicated 0-D sharding for the scalar outputs (loss / grad stats). On TT
    # the partitioner otherwise infers a rank-2 sharding for these scalars
    # (derived from the data/vocab-sharded reductions they come from), which
    # fails compilation with "Output sharding shape (2) doesn't match the
    # output shape (0)". Pinning them replicated keeps the 0-D shape consistent.
    scalar_sharding = NamedSharding(mesh, PartitionSpec()) if mesh is not None else None

    def _pin_scalar(x):
        if scalar_sharding is None:
            return x
        return jax.lax.with_sharding_constraint(x, scalar_sharding)

    def loss_fn(lora_params, frozen_state, input_ids, labels, attention_mask):
        model = nnx.merge(graphdef, lora_params, frozen_state)
        logits = forward(model, input_ids, attention_mask)
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

        #Worakaround for the gradient barrier issue (possible)
        grads = jax.tree.map(
            lambda grad, sharding: jax.lax.with_sharding_constraint(grad, sharding),
            grads,
            lora_shardings,
        )

        updates, new_opt = tx.update(grads, opt_state, lora_params)
        new_params = optax.apply_updates(lora_params, updates)
        new_params = jax.tree.map(jax.lax.optimization_barrier, new_params)
        leaves = jax.tree.leaves(grads)
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
        grad_max = jnp.max(jnp.stack([jnp.max(jnp.abs(g)) for g in leaves]))
        loss = _pin_scalar(loss)
        grad_norm = _pin_scalar(grad_norm)
        grad_max = _pin_scalar(grad_max)
        return new_params, new_opt, loss, {"grad_norm": grad_norm, "grad_max": grad_max}

    return fused_train_step


def create_eval_step_fn(
    graphdef: nnx.GraphDef,
    mesh=None,
    model_axis: str = "model",
    embedding_row_sharded: bool = False,
) -> Callable:
    """JIT-compiled evaluation step returning a scalar loss.

    Uses the vocab-parallel CE when the vocab dim is sharded across
    model_axis, else the TT-safe clamped CE. A single fixed-signature JIT
    (one output, no per-step shape changes) avoids alternating between
    distinct JITs, which previously triggered MeshDevice migration crashes
    on TT (tt-xla#1993, tt-xla#4809).

    embedding_row_sharded selects the input embedding path and must match the
    value passed to create_fused_train_step_fn. See _make_forward_fn.

    Returns eval_step(lora_params, frozen_state, input_ids, labels,
    attention_mask) -> loss (scalar). Predictions are deliberately not
    emitted; see the note inside eval_step.
    """

    cross_entropy = _make_loss_fn(mesh, model_axis)
    forward = _make_forward_fn(mesh, model_axis, embedding_row_sharded)

    def eval_step(lora_params, frozen_state, input_ids, labels, attention_mask):
        model = nnx.merge(graphdef, lora_params, frozen_state)
        logits = forward(model, input_ids, attention_mask)
        # Loss-only single output. We intentionally do NOT also return argmax
        # predictions here: with vocab-parallel logits the loss is produced by a
        # shard_map (manual sharding) while an argmax over the sharded vocab axis
        # is GSPMD auto-sharded, and emitting both leaves the TT flatbuffer with
        # more outputs than collected output shardings ("2 outputs vs 1
        # m_output_shardings"). Keeping a single output also preserves the
        # single fixed-signature eval JIT that avoids MeshDevice re-init crashes
        # (tt-xla#1993, tt-xla#4809).
        return cross_entropy(logits, labels)

    # Pin the scalar loss to a replicated 0-D sharding. Without this TT infers a
    # rank-2 sharding for the scalar (from the data/vocab-sharded reductions it
    # is built from) and fails with "Output sharding shape (2) doesn't match the
    # output shape (0)".
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
    """Run evaluation on validation batches and return average loss.

    jit_eval_step returns a single scalar loss from a fixed-signature JIT.
    When a tokenizer is provided, the first num_examples batches also
    transfer input_ids/labels to CPU and display them via show_predictions
    (input / target / loss; predicted-token text and accuracy are omitted
    since the eval JIT no longer emits argmax predictions). We never
    alternate between distinct eval JITs, which is what previously triggered
    MeshDevice migration crashes on TT (tt-xla#1993, tt-xla#4809).
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
