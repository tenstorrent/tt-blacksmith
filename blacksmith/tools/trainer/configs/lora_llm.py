# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from blacksmith.tools.trainer.configs.base import TrainerConfig


class LoraLLMConfig(TrainerConfig):
    """
    Configuration for :class:`~blacksmith.tools.trainer.strategies.lora_llm_trainer.LoraLLMTrainer`.

    Extends :class:`TrainerConfig` with the LoRA-specific fields needed to
    finalize the LoRA LLM trainer (consumed by ``get_model``, ``get_dataset``
    and the forward/loss computation).
    """

    # Model settings
    max_length: int = Field(default=128, gt=0)

    # Training hyperparameters
    ignored_index: int = Field(default=-100)  # Label id ignored by the cross-entropy loss.

    # LoRA setup
    lora_r: int = Field(default=8, gt=0)
    lora_alpha: int = Field(default=16, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])
    lora_task_type: str = Field(default="CAUSAL_LM")
