# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from blacksmith.tools.configs import TrainingConfig as BaseTrainingConfig


class TrainingConfig(BaseTrainingConfig):
    model_name: str = Field(default="meta-llama/Llama-3.2-1B")

    # LoRA setup
    lora_r: int = Field(default=4, gt=0)
    lora_alpha: int = Field(default=8, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_task_type: str = Field(default="CAUSAL_LM")
