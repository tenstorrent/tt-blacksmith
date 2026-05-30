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

    # NOTE: blacksmith/tools/cli.py merges the test_config block via a shallow
    # `dict |=`, so a per-test yaml that sets `test_config:` replaces this whole
    # dict rather than merging into it. If you add fields here that callers rely
    # on as defaults, either give them sensible behavior when missing or update
    # cli.py to deep-merge.
    model_config = ConfigDict(extra="forbid")

    max_steps_per_epoch: Optional[int] = Field(
        default=None,
        description="Maximum number of batches to process per epoch.",
    )
