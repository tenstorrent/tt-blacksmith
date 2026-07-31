# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

import torch
from pydantic import BaseModel, Field, model_validator

from blacksmith.tools.configs import (
    CheckpointConfig,
    CustomDatasetConfig,
    LoggingConfig,
    MetricsConfig,
)

TORCH_DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}


class TrainerConfig(BaseModel):
    """
    Base trainer configuration.

    Holds fields that are common to all trainer configurations.
    """

    # Dataset settings
    dataset_id: str

    # Model settings
    model_name: str
    dtype: str

    # Training hyperparameters
    learning_rate: float = Field(ge=0.0)
    batch_size: int = Field(gt=0)
    num_epochs: int = Field(gt=0)
    optim: str
    weight_decay: float = Field(ge=0.0)
    gradient_accumulation_steps: int = Field(gt=0)
    training_model_type: str  # [lora, adapters]

    # Validation settings
    val_steps_freq: int = Field(gt=0)

    # Logging / metrics / checkpointing settings (nested sub-configs).
    logging: LoggingConfig
    metrics: MetricsConfig
    checkpoint: CheckpointConfig

    # Custom dataset settings.
    custom_dataset: Optional[CustomDatasetConfig] = Field(default=None)

    @model_validator(mode="after")
    def check_custom_dataset(self) -> "TrainerConfig":
        if self.dataset_id == "custom" and self.custom_dataset is None:
            raise ValueError("`custom_dataset` is required when dataset_id='custom'")
        if self.dataset_id != "custom" and self.custom_dataset is not None:
            raise ValueError("`custom_dataset` is not available when dataset_id!='custom'")
        return self

    # Reproducibility settings
    framework: str
    seed: int
    deterministic: bool

    # Device / sharding settings
    use_tt: bool = Field(default=True)
    mesh_shape: Optional[list[int]] = Field(default=None)
    mesh_axis_names: Optional[list[str]] = Field(default=None)
    input_sharding_dim: Optional[str] = Field(default=None)
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)

    def torch_dtype(self) -> torch.dtype:
        try:
            return TORCH_DTYPES[self.dtype]
        except KeyError as e:
            raise ValueError(f"Unsupported dtype {self.dtype!r}; expected one of {sorted(TORCH_DTYPES)}") from e
