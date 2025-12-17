# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from pydantic import BaseModel, Field
from blacksmith.tools.device_manager import ParallelStrategy


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="mnist")
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    dtype: str = Field(default="torch.bfloat16")

    # Model settings
    model_name: str = Field(default="MNISTLinear")
    input_size: int = Field(default=784, gt=0)
    hidden_size: int = Field(default=512, gt=0)
    output_size: int = Field(default=10, gt=0)
    bias: bool = Field(default=False)

    # Training hyperparameters
    learning_rate: float = Field(default=0.01, gt=0)
    batch_size: int = Field(default=256, gt=0)
    num_epochs: int = Field(default=16, gt=0)
    train_log_steps: int = Field(default=100, gt=0)
    val_log_epochs: int = Field(default=5, gt=0)

    # Loss and optimization
    loss_fn: str = Field(default="torch.nn.MSELoss")
    optim: str = Field(default="sgd")

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="blacksmith-mnist")
    wandb_run_name: str = Field(default="mnist-linear")
    wandb_tags: list[str] = Field(default_factory=lambda: ["tt-xla", "model:torch", "plugin", "wandb"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=100)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=100)
    epoch_freq: int = Field(default=5)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")
    checkpoint_path: str = Field(default="")
    checkpoint_metric: str = Field(default="val/loss")
    checkpoint_metric_mode: str = Field(default="min")
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=1, ge=0)
    save_strategy: str = Field(default="epoch")
    project_dir: str = Field(default="blacksmith/experiments/torch/mnist")
    save_optim: bool = Field(default=False)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=23)
    deterministic: bool = Field(default=False)

    # Device settings
    parallelism_strategy: ParallelStrategy  # [single, data_parallel, tensor_parallel]
    mesh_shape: str = Field(default="8,1")  # Used if parallelism_strategy != single
    tp_sharding_specs: dict[str, tuple[Optional[str], Optional[str]]] = Field(
        default_factory=dict
    )  # Used for model tp sharding

    # Other settings
    device: str = Field(default="TT")
    experiment_name: str = Field(default="torch-mnist")
    output_dir: str = Field(default="experiments/results/mnist")
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)
