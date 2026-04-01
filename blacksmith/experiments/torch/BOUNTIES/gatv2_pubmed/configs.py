# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    # Model settings
    model_name: str = Field(default="GATv2")
    dataset_name: str = Field(default="PubMed")
    dataset_root: str = Field(default="./data")

    # GATv2 architecture
    in_channels: int = Field(default=500)
    hidden_channels: int = Field(default=8)
    out_channels: int = Field(default=3)
    heads: int = Field(default=8)
    dropout: float = Field(default=0.6)

    # Training hyperparameters
    learning_rate: float = Field(default=0.005, gt=0)
    weight_decay: float = Field(default=5e-4)
    num_epochs: int = Field(default=300, gt=0)
    patience: int = Field(default=50)
    val_freq: int = Field(default=1)

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=False)
    wandb_project: str = Field(default="gatv2-pubmed")
    wandb_run_name: str = Field(default="tt-gatv2-pubmed")
    wandb_tags: list[str] = Field(default_factory=lambda: ["gatv2", "pubmed", "node-classification"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=100)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=1)
    epoch_freq: int = Field(default=50)
    val_steps_freq: int = Field(default=1)

    # Checkpoint settings
    project_dir: str = Field(default="blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed")
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")
    checkpoint_path: str = Field(default="")
    checkpoint_metric: str = Field(default="val/accuracy")
    checkpoint_metric_mode: str = Field(default="max")
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=1, ge=0)
    save_strategy: str = Field(default="epoch")
    save_optim: bool = Field(default=False)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=42)
    deterministic: bool = Field(default=True)

    # Device settings
    use_tt: bool = Field(default=False)
    mesh_shape: Optional[list[int]] = Field(default=None)
    mesh_axis_names: Optional[list[str]] = Field(default=None)
    scatter_cpu_fallback: bool = Field(default=False)

    # Other settings
    framework: str = Field(default="pytorch")
