# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field


class LoraConfig(BaseModel):
    """
    Reusable LoRA settings.

    Nested optionally under ``TrainerConfig`` (like metrics / checkpoint) so any
    trainer can enable parameter-efficient fine-tuning by providing this block,
    and run a full fine-tune by leaving it out.
    """

    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    target_modules: list[str]
    task_type: str
