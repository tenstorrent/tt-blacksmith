# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
from typing import Optional

import jax
import optax
from easydel import AutoEasyDeLModelForCausalLM
from flax import nnx
from jax.typing import DTypeLike
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from blacksmith.tools.jax.device_manager import JaxDeviceManager
from blacksmith.tools.templates.configs import TrainingConfig

logger = logging.getLogger(__name__)


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

    load_kwargs: dict = {"dtype": dtype}
    if config_overrides:
        load_kwargs["config_kwargs"] = config_overrides

    axis_size = device_manager.easydel_load_axis_size()
    mesh_name = list(device_manager.mesh.shape.keys())[0]

    load_kwargs["sharding_axis_dims"] = (axis_size,)
    load_kwargs["sharding_axis_names"] = (mesh_name,)
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


def apply_lora(
    model: nnx.Module,
    *,
    rank: int,
    pattern: str,
    on_cpu: bool = True,
    verbose: bool = False,
) -> nnx.Module:
    """Apply LoRA adapters to layers matching pattern, optionally under a CPU context.

    Args:
        model: An EasyDel NNX model.
        rank: LoRA rank.
        pattern: Regex matching layer names to adapt.
        on_cpu: Force CPU context (needed on TT to avoid eager transfers).
        verbose: Print matched layers.

    Returns:
        The model with LoRA layers injected in-place.
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

    Wraps in optax.MultiSteps when config.gradient_accumulation_steps > 1.

    Args:
        config: Training config with LR, warmup, and accumulation fields.
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

    base_optimizer = optax.adamw(learning_rate=schedule)

    accum = config.gradient_accumulation_steps
    if accum > 1:
        optimizer = optax.MultiSteps(base_optimizer, every_k_schedule=accum)
    else:
        optimizer = base_optimizer

    return optimizer, schedule
