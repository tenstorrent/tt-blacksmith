# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

from pydantic import BaseModel
from torchtune.models.llama3_2 import lora_llama3_2_1b
from torchtune.modules.peft._utils import get_adapter_params, set_trainable_params

from thomas.models.torch.dtypes import TorchDType


class TorchTuneModelConfig(BaseModel):
    model_id: str
    dtype: TorchDType
    rank: int
    alpha: int
    lora_attn_modules: List[str]


def load_torch_tune_model(config: TorchTuneModelConfig):
    model = lora_llama3_2_1b(lora_attn_modules=config.lora_attn_modules, lora_rank=config.rank, lora_alpha=config.alpha)
    lora_params = get_adapter_params(model)
    set_trainable_params(model, lora_params)

    return model
