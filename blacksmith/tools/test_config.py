# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for limiting training runs during testing."""
from typing import Optional
from pydantic import BaseModel, Field


class TestConfig(BaseModel):
    """Configuration for test mode to limit training duration.

    This config is used during pytest runs to speed up tests by limiting
    the number of batches processed per epoch.
    """

    max_steps_per_epoch: Optional[int] = Field(
        default=None,
        description="Maximum number of batches to process per epoch.",
    )

    class Config:
        extra = "forbid"  # Prevent typos in config files
