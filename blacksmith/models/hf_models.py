# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import warnings

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from blacksmith.configs import TrainingConfig


def get_model(config: TrainingConfig, device: torch.device):
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
    # Apply LoRA if rank is specified
    if config.training_model_type == "lora":
        model = _apply_lora(model, config)
    elif config.training_model_type == "adapters":
        _apply_adapters(model, config)
    else:
        warnings.warn(
            f"Unknown training_model_type '{config.training_model_type}'; "
            "falling back to full fine-tuning (all parameters trainable)."
        )
        for param in model.parameters():
            param.requires_grad = True

    model.to(config.torch_dtype())
    if config.use_tt:
        model.to(device)

    if config.use_tt and getattr(config, "weight_dtype_overrides", None):
        raise ValueError(
            "weight_dtype_overrides is tt-xla only; use `experimental_weight_dtype` (bfp_bf8 | bfp_bf4) on tt-crank."
        )

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
