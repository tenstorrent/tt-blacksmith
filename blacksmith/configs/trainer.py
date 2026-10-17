# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

from blacksmith.configs.checkpoint import CheckpointConfig
from blacksmith.configs.compile import CompileConfig
from blacksmith.configs.dataset import CustomDatasetConfig
from blacksmith.configs.logging import LoggingConfig
from blacksmith.configs.lora import LoraConfig
from blacksmith.configs.metrics import MetricsConfig


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
    weight_decay: float = Field(ge=0.0)
    gradient_accumulation_steps: int = Field(gt=0)

    # Validation settings. 0 disables validation (no initial or periodic pass).
    val_steps_freq: int = Field(ge=0)

    # Logging / metrics / checkpointing settings (nested sub-configs).
    logging: LoggingConfig
    metrics: MetricsConfig
    checkpoint: CheckpointConfig

    # LoRA settings (nested, optional). When unset the trainer does a full fine-tune.
    lora: Optional[LoraConfig] = Field(default=None)

    # Custom dataset settings.
    custom_dataset: Optional[CustomDatasetConfig] = Field(default=None)

    @model_validator(mode="after")
    def check_custom_dataset(self) -> "TrainerConfig":
        if self.dataset_id == "custom" and self.custom_dataset is None:
            raise ValueError("`custom_dataset` is required when dataset_id='custom'")
        return self

    # Reproducibility settings
    seed: int
    deterministic: bool

    # Device / sharding settings
    use_tt: bool = Field(default=True)
    mesh_shape: Optional[list[int]] = Field(default=None)
    mesh_axis_names: Optional[list[str]] = Field(default=None)
    input_sharding_dim: Optional[str] = Field(default=None)
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)

    # Compile options (see CrankDeviceManager.compile_options).
    compile: CompileConfig = Field(default_factory=CompileConfig)
