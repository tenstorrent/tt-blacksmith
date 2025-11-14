# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AlbertModel
from peft import LoraConfig, get_peft_model

from blacksmith.experiments.torch.llama.configs import TrainingConfig


def get_model(config: TrainingConfig, device: torch.device):
    # This will be replaced with forge models loader, we should add adapter functions to modify the model as needed

    # Load a model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, use_cache=config.gradient_checkpointing)

    # Apply training specific modifications
    # Apply LoRA if rank is specified
    if config.lora_r > 0:
        model = _apply_lora(model, config)

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


class AlbertWithMLP(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.encoder = AlbertModel.from_pretrained(config.model_name)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Simple MLP - NO dropout
        self.classifier = nn.Sequential(
            nn.Linear(768, config.mlp_hidden_dim), nn.GELU(), nn.Linear(config.mlp_hidden_dim, config.num_labels)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        return logits


def get_albert_model(config: TrainingConfig, device: torch.device):
    model = AlbertWithMLP(config)
    model.to(eval(config.dtype))
    model.to(device)

    return model
