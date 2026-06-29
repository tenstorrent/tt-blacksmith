# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from blacksmith.tools.configs import CheckpointConfig, LoggingConfig


class TrainerConfig(BaseModel):
    """
    Base trainer configuration.

    Holds fields that are common to all trainer configurations.
    """

    # Dataset settings
    dataset_id: str = Field(default="dataset_id")

    # Model settings
    model_name: str = Field(default="model_name")
    dtype: str = Field(default="dtype")

    # Training hyperparameters
    learning_rate: float = Field(default=2e-5, gt=0)
    batch_size: int = Field(default=8, gt=0)
    num_epochs: int = Field(default=1, gt=0)
    optim: str = Field(default="adamw_torch")
    weight_decay: float = Field(default=0.0, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    gradient_checkpointing: bool = Field(default=False)
    training_type: str = Field(default="lora")

    # Validation settings
    val_steps_freq: int = Field(default=25, ge=1)

    # Logging / checkpointing settings (nested sub-configs).
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    # Reproducibility settings
    framework: str = Field(default="pytorch")
    seed: int = Field(default=23)
    deterministic: bool = Field(default=False)

    # Device / sharding settings
    use_tt: bool = Field(default=True)
    mesh_shape: Optional[list[int]] = Field(default=None)
    mesh_axis_names: Optional[list[str]] = Field(default=None)
    input_sharding_dim: Optional[str] = Field(default=None)
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)
