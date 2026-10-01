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

    # NOTE: blacksmith/tools/cli.py deep-merges the per-test YAML into the
    # experiment config, so a `test_config:` block here merges into the
    # PYTEST_CURRENT_TEST defaults rather than replacing them.
    model_config = ConfigDict(extra="forbid")

    max_steps_per_epoch: Optional[int] = Field(
        default=None,
        description="Maximum number of batches to process per epoch.",
    )
    cpu_sample_rng: bool = Field(
        default=False,
        description=(
            "TT RNG is not reliably seedable." "Randomness in tests needs to be reproducible on CPU at least."
        ),
    )
