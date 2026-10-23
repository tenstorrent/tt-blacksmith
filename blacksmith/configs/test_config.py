# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TestConfig(BaseModel):
    """
    Test-mode settings that keep a golden-file run short (as in the tt-xla tree).

    Set from a test overlay YAML (`test_config:` block); never needed for a real run.
    """

    model_config = ConfigDict(extra="forbid")

    max_steps_per_epoch: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum number of batches per epoch, for every split.",
    )
