# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from thomas.models.torch.dtypes import TorchDType


class LoraModelConfig(BaseModel):
    model_id: str
    dtype: TorchDType
    rank: int
    alpha: int


def load_hf_model(config: LoraModelConfig):
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=config.dtype)

    lora_config = LoraConfig(r=config.rank, lora_alpha=config.alpha)
    model = get_peft_model(model, lora_config)

    return model
