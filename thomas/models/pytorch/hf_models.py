# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from thomas.experiments.pytorch.configs import TrainingConfig


def get_model(config: TrainingConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=eval(config.dtype), use_cache=config.gradient_checkpointing
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=config.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)

    return model
