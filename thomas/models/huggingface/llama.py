# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import wandb
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from thomas.experiments.huggingface.configs import TrainingConfig


class LlamaLoraModel:
    def __init__(self, config: TrainingConfig):
        self.config = config

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, torch_dtype=eval(self.config.dtype), use_cache=self.config.gradient_checkpointing
        )

        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            target_modules="all-linear",
        )
        self.model = get_peft_model(self.model, lora_config)

    def log_model_params(self):
        total_params = sum([p.numel() for p in self.model.parameters()])
        trainable_params = sum([p.numel() for p in self.model.parameters() if p.requires_grad])

        model_logs = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_param_percentage": round((100.0 * trainable_params / total_params), 2),
        }
        wandb.config.update(model_logs)
