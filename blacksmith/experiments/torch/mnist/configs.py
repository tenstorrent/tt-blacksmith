# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

from pydantic import Field

from blacksmith.tools.templates.configs import TrainingConfig as BaseTrainingConfig
from blacksmith.tools.test_config import TestConfig


class TrainingConfig(BaseTrainingConfig):
    # Dataset settings
    dataset_id: str = Field(default="mnist")
    train_ratio: float = Field(default=0.8, gt=0, lt=1)

    # Model settings (for MNISTLinear)
    model_name: str = Field(default="MNISTLinear")
    input_size: int = Field(default=784, gt=0)
    hidden_size: int = Field(default=512, gt=0)
    output_size: int = Field(default=10, gt=0)
    bias: bool = Field(default=False)

    # CNN model settings (for MNISTCNN)
    conv1_channels: int = Field(default=32, gt=0)
    conv2_channels: int = Field(default=64, gt=0)
    kernel_size: int = Field(default=3, gt=0)
    stride: int = Field(default=1, gt=0)
    fc1_size: int = Field(default=128, gt=0)
    dropout1_rate: float = Field(default=0.25, ge=0, le=1)
    dropout2_rate: float = Field(default=0.5, ge=0, le=1)

    # Training hyperparameters
    learning_rate: float = Field(default=0.01, gt=0)
    batch_size: int = Field(default=256, gt=0)
    num_epochs: int = Field(default=16, gt=0)

    # Loss and optimization
    loss_fn: str = Field(default="torch.nn.MSELoss")
    optim: str = Field(default="sgd")

    # Logging settings
    wandb_project: str = Field(default="blacksmith-mnist")
    wandb_run_name: str = Field(default="mnist-linear")
    wandb_tags: list[str] = Field(default_factory=lambda: ["tt-xla", "model:torch", "plugin", "wandb"])
    wandb_log_freq: int = Field(default=100)
    steps_freq: int = Field(default=100)
    val_steps_freq: int = Field(default=100)
    epoch_freq: int = Field(default=5)

    # Checkpoint settings
    checkpoint_metric: str = Field(default="val/loss")
    keep_best_n: int = Field(default=1, ge=0)
    project_dir: str = Field(default="blacksmith/experiments/torch/mnist")

    # Device settings
    mesh_shape: Optional[list[int]] = Field(default=None)  # Use None for single device, [x,y] for 2D mesh.
    mesh_axis_names: Optional[list[str]] = Field(
        default=None
    )  # Use None for single device, ["data", "model"] for 2D mesh.
    input_sharding_dim: Optional[str] = Field(
        default=None
    )  # If defined, we will shard inputs along this mesh axis dimension.
    # Tensor parallelism sharding patterns (regex pattern based - matches module names).
    # Format: List of tuples (regex_pattern, sharding_spec_tuple).
    model_sharding_patterns: Optional[List[Tuple[str, Tuple[Optional[str], ...]]]] = Field(default=None)

    # Other settings
    device: str = Field(default="TT")
    experiment_name: str = Field(default="torch-mnist")
    output_dir: str = Field(default="experiments/results/mnist")
    test_config: Optional[TestConfig] = Field(default=None)
