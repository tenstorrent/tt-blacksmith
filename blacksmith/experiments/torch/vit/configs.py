# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional, Tuple

from pydantic import Field

from blacksmith.tools.templates.configs import TrainingConfig as BaseTrainingConfig
from blacksmith.tools.test_config import TestConfig


class TrainingConfig(BaseTrainingConfig):
    # Dataset settings
    dataset_id: str = Field(default="stanfordcars")
    image_size: int = Field(default=224, gt=0)
    image_mean: list[float] = Field(default=[0.5, 0.5, 0.5])
    image_std: list[float] = Field(default=[0.5, 0.5, 0.5])

    # Model settings
    model_name: str = Field(default="google/vit-base-patch16-224")
    ignored_index: int = Field(default=-100)

    # Training hyperparameters
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=10, gt=0)
    num_epochs: int = Field(default=8, gt=0)

    # Loss
    loss_fn: str = Field(default="torch.nn.CrossEntropyLoss")

    # Logging settings
    wandb_project: str = Field(default="vit-finetuning")
    wandb_run_name: str = Field(default="tt-vit-stanfordcars")
    steps_freq: int = Field(default=10)
    val_steps_freq: int = Field(default=50)

    # Checkpoint settings
    project_dir: str = Field(default="blacksmith/experiments/torch/vit")

    # Device settings
    mesh_shape: Optional[list[int]] = Field(default=None)  # Use None for single device, [x,y] for 2D mesh.
    mesh_axis_names: Optional[list[str]] = Field(
        default=None
    )  # Use None for single device, ["data", "model"] for 2D mesh.
    input_sharding_dim: Optional[str] = Field(
        default=None
    )  # If defined, we will shard inputs along this mesh axis dimension.
    # Model sharding patterns (regex pattern based - matches module names).
    # Format: List of tuples (regex_pattern, sharding_spec_tuple).
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)

    # LoRA setup
    lora_r: int = Field(default=4, gt=0)
    lora_alpha: int = Field(default=8, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])
    lora_dropout: float = Field(default=0.1, ge=0, le=1)

    # Other settings
    test_config: Optional[TestConfig] = Field(default=None)
