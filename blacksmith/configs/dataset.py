# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class CustomDatasetConfig(BaseModel):
    """
    Additional config in case of custom datasets.
    Train and validation sets should be loaded from separate files.
    """

    model_config = ConfigDict(extra="forbid")

    file_type: str = Field(default="json")
    train_dataset_path: Optional[str] = Field(default=None)
    val_dataset_path: Optional[str] = Field(default=None)

    # Define template type (Alpaca-style, chat, etc)
    template: str = Field(default="alpaca")

    column_mapping: Optional[Dict[str, str]] = Field(default=None)
