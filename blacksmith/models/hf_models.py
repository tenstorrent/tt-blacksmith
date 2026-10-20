# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM

from blacksmith.configs import TrainerConfig
from blacksmith.models.workarounds import CAUSAL_SDPA, flatten_linear_inputs

TORCH_DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}


def torch_dtype(dtype: str) -> torch.dtype:
    """Map a config dtype string (e.g. ``"torch.bfloat16"``) to a ``torch.dtype``."""
    try:
        return TORCH_DTYPES[dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(TORCH_DTYPES)}") from e


def get_model(config: TrainerConfig, device: torch.device):
    # Before loading, so an unsupported dtype fails without loading any weights.
    dtype = torch_dtype(config.dtype)

    # Load a model
    load_kwargs = {
        "use_cache": False,
        "low_cpu_mem_usage": True,
        "attn_implementation": CAUSAL_SDPA,
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

    model.to(dtype)
    if config.use_tt:
        flatten_linear_inputs(model)
        model.to(device)

    return model


def _apply_lora(model, lora_config):
    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        task_type=lora_config.task_type,
    )

    return get_peft_model(model, peft_config)
