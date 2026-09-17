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
from blacksmith.tools.templates.configs import MeshConfig
from blacksmith.tools.test_config import TestConfig

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

    # Validation settings. 0 disables validation (no initial or periodic pass).
    val_steps_freq: int = Field(ge=0)

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
        return self

    # Reproducibility settings
    framework: str
    seed: int
    deterministic: bool

    # Device / sharding settings. tt-crank reads `mesh` (see MeshConfig); the tt-xla quartet
    # below is accepted so old YAMLs validate, but the sharding rules live in DeviceManager.
    use_tt: bool = Field(default=True)
    mesh: Optional[MeshConfig] = Field(default=None)
    mesh_shape: Optional[list[int]] = Field(default=None)
    mesh_axis_names: Optional[list[str]] = Field(default=None)
    input_sharding_dim: Optional[str] = Field(default=None)
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)

    # tt-crank compile options (forwarded per torch.compile call, see DeviceManager.compile_options).
    optimization_level: int = Field(default=1, ge=0, le=2)
    enable_const_eval: bool = Field(default=False)
    enable_trace: bool = Field(default=False)
    math_fidelity: str = Field(default="HiFi4")  # [LoFi, HiFi2, HiFi3, HiFi4]
    fp32_dest_acc_en: bool = Field(default=True)
    experimental_weight_dtype: Optional[str] = Field(default=None)  # [BfpBf8, BfpBf4]

    # pytest step-limiting; set by generate_config under PYTEST_CURRENT_TEST.
    test_config: Optional[TestConfig] = Field(default=None)

    def torch_dtype(self) -> torch.dtype:
        try:
            return TORCH_DTYPES[self.dtype]
        except KeyError as e:
            raise ValueError(f"Unsupported dtype {self.dtype!r}; expected one of {sorted(TORCH_DTYPES)}") from e
