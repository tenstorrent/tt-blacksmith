# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Self

from pydantic import Field, model_validator

from blacksmith.tools.templates.configs import TrainingConfig
from blacksmith.tools.test_config import TestConfig


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
    # Keep sampling defaults at the fixed-shape sizes validated on Wormhole.
    # The stock CPU YAML explicitly opts into its larger sampling workload.
    batch_size: int = Field(default=32, gt=0)
    num_epochs: int = Field(default=30, gt=0)
    num_neighbors: list[int] = Field(default=[5, 3])
    val_batch_size: int = Field(default=32, gt=0)

    # Logging
    use_wandb: bool = Field(default=False)
    wandb_project: str = Field(default="graphsage-reddit")
    wandb_run_name: str = Field(default="graphsage-reddit-cpu")
    wandb_tags: list[str] = Field(default_factory=lambda: ["graphsage", "reddit", "cpu"])

    # Checkpoint
    checkpoint_metric: str = Field(default="val/acc")
    checkpoint_metric_mode: str = Field(default="max")
    epoch_freq: int = Field(default=5)
    project_dir: str = Field(default="blacksmith/experiments/torch/BOUNTIES/graphsage_reddit")

    # Device
    use_tt: bool = Field(default=False)
    use_spmm: bool = Field(default=False)
    static_shapes: bool = Field(default=False)

    # Testing
    test_config: Optional[TestConfig] = Field(default=None)

    @model_validator(mode="after")
    def validate_graphsage_execution(self) -> Self:
        """Validate sampling depth and the fixed-shape TT execution contract."""
        if len(self.num_neighbors) != 2:
            raise ValueError("two-layer GraphSAGE requires exactly two neighbor fanouts")
        if self.static_shapes and any(fanout < 0 for fanout in self.num_neighbors):
            raise ValueError("static shapes require finite non-negative fanouts")
        if self.use_tt and (not self.use_spmm or not self.static_shapes):
            raise ValueError("TT execution requires use_spmm=true and static_shapes=true")
        return self
