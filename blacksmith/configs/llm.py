# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from blacksmith.configs.trainer import TrainerConfig


class LLMTrainerConfig(TrainerConfig):
    """
    Configuration for the LLM trainer.

    LoRA is opt-in via the nested ``lora`` block on ``TrainerConfig``; leaving it
    out runs a full fine-tune.
    """

    # Model settings
    max_length: int = Field(gt=0)
