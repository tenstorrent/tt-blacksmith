# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from blacksmith.tools.templates.configs import TrainingConfig


def get_model(config: TrainingConfig, device: torch.device):
    # This will be replaced with forge models loader, we should add adapter functions to modify the model as needed

    # Load a model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, use_cache=config.gradient_checkpointing)

    # Apply training specific modifications
    # Apply LoRA if rank is specified
    if config.training_type == "lora":
        model = _apply_lora(model, config)
    elif config.training_type == "adapters":
        _apply_adapters(model, config)
    else:
        raise ValueError(f"Invalid training type: {config.training_type}")

    model.to(eval(config.dtype))
    model.to(device)

    return model


def _apply_lora(model, config: TrainingConfig):
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        task_type=config.lora_task_type,
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
        adapter = make_adapter(original_layer_output.out_features, original_layer_output.out_features, config)
        new_layer_output = torch.nn.Sequential(original_layer_output, *adapter)
        model.model.layers[block_idx].self_attn.o_proj = new_layer_output

        #### Insert second adapter
        original_layer_output = model.model.layers[block_idx].mlp.down_proj
        adapter = make_adapter(original_layer_output.out_features, original_layer_output.out_features, config)
        new_layer_output = torch.nn.Sequential(original_layer_output, *adapter)
        model.model.layers[block_idx].mlp.down_proj = new_layer_output

    return model


def make_adapter(in_dim, out_dim, config: TrainingConfig):
    bottleneck_dim = config.adapter_bottleneck_dim
    adapter_layers = torch.nn.Sequential(
        torch.nn.Linear(in_dim, bottleneck_dim),
        eval(config.adapter_non_linearity)(),
        torch.nn.Linear(bottleneck_dim, out_dim),
    )

    return adapter_layers
