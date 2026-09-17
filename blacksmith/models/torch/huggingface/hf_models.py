# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace causal-LM loader with LoRA / adapters applied.

Same API as the tt-xla tree; only the device/compile handling is tt-crank's.
"""
import warnings

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from blacksmith.tools.templates.configs import TrainingConfig


def get_model(config: TrainingConfig, device: torch.device, compile_model: bool = True):
    # This will be replaced with forge models loader, we should add adapter functions to modify the model as needed

    # Load a model
    load_kwargs = {
        "use_cache": False,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        # Stream weights directly to GPU to avoid CPU RAM spikes during load.
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)

    # Apply training specific modifications
    training_model_type = getattr(config, "training_model_type", "lora")
    if training_model_type == "lora":
        model = _apply_lora(model, config)
    elif training_model_type == "adapters":
        _apply_adapters(model, config)
    else:
        warnings.warn(
            f"Unknown training_model_type '{training_model_type}'; "
            "falling back to full fine-tuning (all parameters trainable)."
        )
        for param in model.parameters():
            param.requires_grad = True

    # Cast on the host first: moving a bf16 model is half the transfer of
    # moving fp32 weights and then casting on device.
    model.to(config.torch_dtype())
    if config.use_tt:
        model.to(device)

    # Per-tensor weight dtype overrides were a tt-xla feature (tt_torch.apply_weight_dtype_overrides).
    # tt-crank only has the compiler-wide `experimental_weight_dtype` compile option.
    if config.use_tt and getattr(config, "weight_dtype_overrides", None):
        raise ValueError(
            "weight_dtype_overrides is tt-xla only; tt-crank has no per-tensor override. "
            "Use `experimental_weight_dtype` (BfpBf8 | BfpBf4) for a compiler-wide default instead."
        )

    if config.use_tt and compile_model:
        # tt-xla wrapped the model with its dynamo knobs (tt_legacy_compile, tt_lazy_execution, ...).
        # tt-crank takes the config's compile options per `torch.compile` call. Experiments that
        # compile forward+loss as one callable pass compile_model=False and compile themselves.
        from blacksmith.tools.device_manager import tt_compile_options

        model = torch.compile(model, backend="tt", options=tt_compile_options(config))

    return model


def _apply_lora(model, config: TrainingConfig):
    # When unfreeze_embeddings is enabled, use modules_to_save to also train
    # the embedding layer alongside LoRA adapters. This is needed for models
    # like Falcon3 that have limited language coverage - unfreezing embeddings
    # allows the model to adapt token representations for unseen languages.
    modules_to_save = None
    if getattr(config, "unfreeze_embeddings", False):
        modules_to_save = ["embed_tokens"]

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        task_type=config.lora_task_type,
        modules_to_save=modules_to_save,
    )

    return get_peft_model(model, lora_config)


def _apply_adapters(model, config: TrainingConfig):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Apply adapters
    if len(config.adapter_layers) == 0:
        adapter_layers = list(range(len(model.model.layers)))
    else:
        adapter_layers = config.adapter_layers

    for block_idx in adapter_layers:
        #### Insert first adapter
        original_layer_output = model.model.layers[block_idx].self_attn.o_proj
        adapted_layer = make_adapted_layer(original_layer_output, config)
        model.model.layers[block_idx].self_attn.o_proj = adapted_layer

        #### Insert second adapter
        original_layer_output = model.model.layers[block_idx].mlp.down_proj
        adapted_layer = make_adapted_layer(original_layer_output, config)
        model.model.layers[block_idx].mlp.down_proj = adapted_layer

    return model


def make_adapted_layer(linear, config: TrainingConfig):
    class ResidualAdapter(nn.Module):
        def __init__(self, linear, bottleneck_dim):
            super().__init__()
            self.linear = linear
            d = linear.out_features

            self.adapter = nn.Sequential(
                nn.Linear(d, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, d),
            )

            # Start as identity
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

        def forward(self, x):
            y = self.linear(x)
            return y + self.adapter(y)

    return ResidualAdapter(linear, config.adapter_bottleneck_dim)
