# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TestConfig(BaseModel):
    """
    Configuration for test mode to limit training duration.

    This config is used during pytest runs to speed up tests by limiting
    the number of batches processed per epoch.
    """

    model_config = ConfigDict(extra="forbid")

    max_steps_per_epoch: Optional[int] = Field(
        default=None,
        description="Maximum number of batches to process per epoch.",
    )
