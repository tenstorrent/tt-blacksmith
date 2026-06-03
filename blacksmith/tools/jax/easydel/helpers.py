# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
from typing import Optional

import jax
import jax.numpy as jnp
import optax
from easydel import AutoEasyDeLModelForCausalLM
from flax import nnx
from flax.nnx.nn.dtypes import promote_dtype
from flax.nnx.nn.lora import LoRA as _FlaxLoRA
from jax.typing import DTypeLike
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from blacksmith.tools.jax.device_manager import JaxDeviceManager
from blacksmith.tools.jax.easydel.partitioning import _path_to_str
from blacksmith.tools.templates.configs import TrainingConfig

logger = logging.getLogger(__name__)


def embedding_is_row_sharded(frozen_state, model_axis: str = "model") -> bool:
    """Return True if the token embedding is sharded along its vocab (row) axis.

    We read the actual placed sharding of the embed_tokens embedding leaf, so
    the result reflects whatever the yaml sharding patterns produced rather than
    any assumption. When the leading (vocab) axis lands on model_axis the input
    lookup must use the vocab-parallel path; otherwise the plain replicated
    lookup is correct and avoids emitting an extra manual computation region.
    """
    flat, _ = jax.tree_util.tree_flatten_with_path(frozen_state)
    for path, leaf in flat:
        if _path_to_str(path).endswith("embed_tokens.embedding.value"):
            spec = getattr(getattr(leaf, "sharding", None), "spec", None)
            return spec is not None and len(spec) >= 1 and spec[0] == model_axis
    return False


def load_tokenizer(
    model_name: str,
) -> PreTrainedTokenizerBase:
    """Load a HuggingFace tokenizer, ensuring a pad token exists."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_easydel_causal_lm(
    model_name: str,
    device_manager: JaxDeviceManager,
    *,
    dtype: DTypeLike,
    mask_max_position_embeddings: Optional[int] = None,
    auto_shard_model: bool = False,
    extra_config_kwargs: Optional[dict] = None,
) -> tuple[nnx.Module, PreTrainedTokenizerBase]:
    """Load an EasyDel causal LM and its tokenizer.

    On TT the model is loaded under a CPU default-device context to avoid
    eager host-to-device transfers during init; the mesh is attached afterwards.

    Args:
        model_name: HuggingFace model identifier.
        device_manager: Initialised JaxDeviceManager.
        dtype: JAX dtype for model weights.
        mask_max_position_embeddings: Override for the model's max_position_embeddings.
        auto_shard_model: Let EasyDel auto-shard during loading.
        extra_config_kwargs: Extra kwargs forwarded to the EasyDel model config.

    Returns:
        (model, tokenizer) tuple.
    """
    config_overrides: dict = {}
    if mask_max_position_embeddings is not None:
        config_overrides["mask_max_position_embeddings"] = mask_max_position_embeddings
    if extra_config_kwargs:
        config_overrides.update(extra_config_kwargs)

    load_kwargs: dict = {"dtype": dtype, "param_dtype": dtype}
    if config_overrides:
        load_kwargs["config_kwargs"] = config_overrides

    load_kwargs["sharding_axis_dims"] = tuple(device_manager.mesh.shape.values())
    load_kwargs["sharding_axis_names"] = tuple(device_manager.mesh.shape.keys())

    load_kwargs["auto_shard_model"] = auto_shard_model

    on_tt = device_manager.device_kind == "tt"
    ctx = jax.default_device(jax.devices("cpu")[0]) if on_tt else contextlib.nullcontext()

    with ctx:
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs,
        )

    model.config.set_model_mesh(device_manager.mesh)
    tokenizer = load_tokenizer(model_name)

    return model, tokenizer


def _lora_call_with_barrier(self, x: jax.Array) -> jax.Array:
    """LoRA forward that isolates the adapter matmul from the base projection.

    Identical to flax's LoRA.__call__ (out = x @ lora_a @ lora_b +
    base_module(x)) but inserts an optimization barrier on the base input so XLA
    cannot fuse the replicated x @ lora_a matmul into the model-sharded base
    projection. Under tensor parallelism that fusion concats replicated lora_a
    columns with the column-parallel base kernel, after which the downstream
    slice_static uses global offsets that overrun the local shard width and
    crash with a TT_FATAL. The barrier forces two independent dots instead.
    """
    x, lora_a, lora_b = promote_dtype((x, self.lora_a[...], self.lora_b[...]), dtype=self.dtype)
    out = x @ lora_a @ lora_b
    if self.base_module is not None:
        if not callable(self.base_module):
            raise ValueError("self.base_module must be callable.")
        # Break the x-sharing that lets XLA concat replicated lora_a columns with
        # the model-sharded base kernel into one fused (and un-sliceable) matmul.
        x_base = jax.lax.optimization_barrier(x)
        out += self.base_module(x_base)
    return out


def apply_lora(
    model: nnx.Module,
    *,
    rank: int,
    pattern: str,
    on_cpu: bool = True,
    verbose: bool = False,
) -> nnx.Module:
    """Apply LoRA adapters to layers matching pattern, optionally under a CPU context.

    Also patches flax's LoRA.__call__ with _lora_call_with_barrier so the
    adapter matmul is not fused into the model-sharded base projection (required
    for column-parallel q/v under tensor parallelism on TT; see that function).

    Args:
        model: An EasyDel NNX model.
        rank: LoRA rank.
        pattern: Regex matching layer names to adapt.
        on_cpu: Force CPU context (needed on TT to avoid eager transfers).
        verbose: Print matched layers.

    Returns:
        The model with LoRA layers injected in-place.
    """
    # Idempotent class patch: isolates the LoRA adapter dot from the base dot.
    """
    if _FlaxLoRA.__call__ is not _lora_call_with_barrier:
        _FlaxLoRA.__call__ = _lora_call_with_barrier
    """
    
    ctx = jax.default_device(jax.devices("cpu")[0]) if on_cpu else contextlib.nullcontext()
    with ctx:
        return model.apply_lora_to_layers(
            lora_rank=rank,
            lora_pattern=pattern,
            verbose=verbose,
        )


def build_optimizer(
    config: TrainingConfig,
    *,
    total_opt_steps: int,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Build an AdamW optimizer with a warmup-cosine-decay schedule.

    When config.max_grad_norm is a positive float, prepends
    optax.clip_by_global_norm so the accumulated gradient is clipped
    right before the AdamW update (same semantics as torch's
    clip_grad_norm_; matches the gpt_oss / distil_bert / nanogpt
    convention in this repo).

    Wraps in optax.MultiSteps when config.gradient_accumulation_steps > 1
    so clipping fires on accumulation-completion steps.

    Args:
        config: Training config with LR, warmup, accumulation, and clipping fields.
        total_opt_steps: Total number of optimizer updates after accumulation.

    Returns:
        (optimizer, schedule) tuple.
    """
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=getattr(config, "warmup_steps", 0),
        decay_steps=total_opt_steps,
        end_value=getattr(config, "end_learning_rate", 0.0),
    )

    transforms: list[optax.GradientTransformation] = []
    max_grad_norm = getattr(config, "max_grad_norm", None)
    if max_grad_norm is not None and max_grad_norm > 0:
        transforms.append(optax.clip_by_global_norm(max_grad_norm))
    transforms.append(optax.adamw(learning_rate=schedule, mu_dtype=jnp.float32, eps=1e-5))
    base_optimizer = optax.chain(*transforms)

    accum = config.gradient_accumulation_steps
    if accum > 1:
        optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=accum)
    else:
        optimizer = base_optimizer

    return optimizer, schedule
