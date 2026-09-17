# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from blacksmith.tools.trainer.configs.base import TrainerConfig


class LoraLLMConfig(TrainerConfig):
    """
    Configuration for the LoRA LLM trainer.
    """

    # Model settings
    max_length: int = Field(gt=0)

    # LoRA setup
    lora_r: int = Field(gt=0)
    lora_alpha: int = Field(gt=0)
    lora_target_modules: list[str]
    lora_task_type: str
