# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_name: str = Field(default="PubMed")
    dataset_root: str = Field(default="blacksmith/experiments/torch/BOUNTIES/gatv2-pubmed/.cache/planetoid")

    # Model settings
    model_name: str = Field(default="GATv2")
    hidden_channels: int = Field(default=8, gt=0)
    heads: int = Field(default=8, gt=0)
    dropout: float = Field(default=0.6, ge=0, le=1)

    # Training hyperparameters
    learning_rate: float = Field(default=0.005, gt=0)
    weight_decay: float = Field(default=5e-4, ge=0)
    num_epochs: int = Field(default=300, gt=0)
    early_stop_patience: int = Field(default=50, gt=0)
    log_interval: int = Field(default=10, gt=0)

    # Reproducibility settings
    seed: int = Field(default=42)
    deterministic: bool = Field(default=True)

    # Logging and outputs
    log_level: str = Field(default="INFO")
    output_dir: str = Field(default="blacksmith/experiments/torch/BOUNTIES/gatv2-pubmed/results")
    project_dir: str = Field(default="blacksmith/experiments/torch/BOUNTIES/gatv2-pubmed")
    run_name: str = Field(default="gatv2-pubmed-cpu-baseline")
    use_wandb: bool = Field(default=False)

    # Device settings
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=False)
    device: str = Field(default="cpu")

