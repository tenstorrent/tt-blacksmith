# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, ConfigDict, Field


class LoraConfig(BaseModel):
    """
    LoRA settings.
    """

    model_config = ConfigDict(extra="forbid")

    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    target_modules: list[str]
    task_type: str
