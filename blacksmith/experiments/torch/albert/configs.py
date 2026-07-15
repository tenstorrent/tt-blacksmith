# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

from pydantic import Field

from blacksmith.tools.templates.configs import TrainingConfig as BaseTrainingConfig
from blacksmith.tools.test_config import TestConfig


class TrainingConfig(BaseTrainingConfig):
    # Dataset settings
    dataset_id: str = Field(default="banking77")

    # Model settings
    model_name: str = Field(default="albert/albert-base-v2")
    num_labels: int = Field(default=2, gt=0)
    mlp_hidden_dim: int = Field(default=256, gt=0)

    # Training hyperparameters
    training_model_type: str = Field(default="lora")  # [lora, adapters]
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)

    # Logging settings
    wandb_project: str = Field(default="albert-finetuning")
    wandb_run_name: str = Field(default="tt-albert-test")

    # Checkpoint settings
    project_dir: str = Field(default="blacksmith/experiments/torch/albert")

    # Device settings
    mesh_shape: Optional[list[int]] = Field(default=None)  # Use None for single device, [x,y] for 2D mesh.
    mesh_axis_names: Optional[list[str]] = Field(
        default=None
    )  # Use None for single device, ["data", "model"] for 2D mesh.
    # Model sharding patterns (regex pattern based - matches module names).
    # Format: List of tuples (regex_pattern, sharding_spec_tuple).
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)
    input_sharding_dim: Optional[str] = Field(
        default=None
    )  # If defined, we will shard inputs along this mesh axis dimension.

    # Other settings
    test_config: Optional[TestConfig] = Field(default=None)
