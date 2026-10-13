# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM

from blacksmith.configs import TrainerConfig
from blacksmith.tools.crank.torch_helpers import torch_dtype


def get_model(config: TrainerConfig, device: torch.device):
    # Load a model
    load_kwargs = {
        "use_cache": False,
        "low_cpu_mem_usage": True,
    }
    if device.type == "cuda":
        # Stream weights directly to GPU to avoid CPU RAM spikes during load.
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **load_kwargs)

    # Apply LoRA when configured; otherwise full fine-tuning (all params trainable).
    if config.lora is not None:
        model = _apply_lora(model, config.lora)
    else:
        for param in model.parameters():
            param.requires_grad = True

    model.to(torch_dtype(config.dtype))
    if config.use_tt:
        model.to(device)

    if config.use_tt and getattr(config, "weight_dtype_overrides", None):
        raise ValueError("weight_dtype_overrides is deprecated; use `experimental_weight_dtype` (bfp_bf8 | bfp_bf4).")

    return model


def _apply_lora(model, lora_config):
    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        target_modules=lora_config.target_modules,
        task_type=lora_config.task_type,
    )

    return get_peft_model(model, peft_config)
