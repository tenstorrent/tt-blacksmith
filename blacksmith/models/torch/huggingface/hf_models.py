# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from peft import LoraConfig, get_peft_model

from blacksmith.experiments.torch.llama.configs import TrainingConfig


class TextModelWrapper(torch.nn.Module):
    def __init__(self, model, text_embedding=None):
        super().__init__()
        self.model = model
        self.text_embedding = text_embedding

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is not None and self.text_embedding is not None:
            inputs_embeds = self.text_embedding(input_ids)
            past_key_values_length = 0
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_ids.shape, inputs_embeds, past_key_values_length
            )
            logits = self.model(attention_mask=causal_attention_mask, inputs_embeds=inputs_embeds).logits
        else:
            logits = self.model(input_ids=input_ids).logits
        return logits


def get_model(config: TrainingConfig):
    model = AutoModelForCausalLM.from_pretrained(config.model_name, use_cache=config.gradient_checkpointing)

    # Applying LoRA to the last half of the layers due to memory constraints
    ltt = range(model.config.num_hidden_layers // 2, model.config.num_hidden_layers)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        layers_to_transform=ltt,
        task_type=config.lora_task_type,
    )
    model = get_peft_model(model, lora_config)
    model.to(eval(config.dtype))

    return model
