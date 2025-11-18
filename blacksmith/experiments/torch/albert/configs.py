# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="mteb/banking77")

    # Model settings
    model_name: str = Field(default="albert/albert-base-v2")
    max_length: int = Field(default=128, gt=0)
    num_labels: int = Field(default=2, gt=0)
    mlp_hidden_dim: int = Field(default=256, gt=0)
    dtype: str = Field(default="torch.bfloat16")

    # Training hyperparameters
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    batch_size: int = Field(default=32, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    gradient_checkpointing: bool = Field(default=False)
    num_epochs: int = Field(default=1, gt=0)
    optim: str = Field(default="adamw_torch")

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="albert-finetuning")
    wandb_run_name: str = Field(default="tt-albert-test")
    wandb_tags: list[str] = Field(default_factory=lambda: ["test"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=25)
    epoch_freq: int = Field(default=1)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")  # [last, best, path]
    checkpoint_path: str = Field(default="")  # path to checkpoint if resume_option is "path"
    checkpoint_metric: str = Field(default="eval/loss")
    checkpoint_metric_mode: str = Field(default="min")  # [min, max]
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=3, ge=0)
    save_strategy: str = Field(default="epoch")
    project_dir: str = Field(default="blacksmith/experiments/torch/albert")
    save_optim: bool = Field(default=False)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=23)
    deterministic: bool = Field(default=False)

    # Other settings
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)
