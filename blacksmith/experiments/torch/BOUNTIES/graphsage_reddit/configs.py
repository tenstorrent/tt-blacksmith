# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from blacksmith.tools.templates.configs import TrainingConfig


class GraphSAGEConfig(TrainingConfig):
    # Dataset
    dataset_id: str = Field(default="Reddit")
    dataset_root: str = Field(default="/tmp/Reddit")

    # Model
    hidden_channels: int = Field(default=256, gt=0)
    dropout: float = Field(default=0.5, ge=0.0, le=1.0)

    # Training
    learning_rate: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=5e-4, ge=0.0)
    batch_size: int = Field(default=512, gt=0)
    num_epochs: int = Field(default=30, gt=0)
    num_neighbors: list[int] = Field(default=[25, 10])
    val_batch_size: int = Field(default=4096, gt=0)

    # Logging
    use_wandb: bool = Field(default=False)
    wandb_project: str = Field(default="graphsage-reddit")
    wandb_run_name: str = Field(default="graphsage-reddit-cpu")
    wandb_tags: list[str] = Field(
        default_factory=lambda: ["graphsage", "reddit", "cpu"]
    )

    # Checkpoint
    checkpoint_metric: str = Field(default="val/acc")
    checkpoint_metric_mode: str = Field(default="max")
    epoch_freq: int = Field(default=5)
    project_dir: str = Field(
        default="blacksmith/experiments/torch/BOUNTIES/graphsage_reddit"
    )

    # Device
    use_tt: bool = Field(default=False)
