# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="tiny-imagenet")
    train_ratio: float = Field(default=0.9, gt=0, lt=1)  # 1 - eval_split
    dtype: str = Field(default="torch.float32")
    data_path: str = Field(default="tiny-imagenet-200")

    # Model settings
    model_name: str = Field(default="alexnet")
    input_size: int = Field(default=256, gt=0)  # original image_size
    hidden_size: int = Field(default=0)  # Not specified for AlexNet
    output_size: int = Field(default=200, gt=0)  # num_classes
    bias: bool = Field(default=True)

    # Training hyperparameters
    learning_rate: float = Field(default=0.001, gt=0)
    batch_size: int = Field(default=256, gt=0)
    num_epochs: int = Field(default=10, gt=0)
    train_log_steps: int = Field(default=100, gt=0)
    val_log_epochs: int = Field(default=1, gt=0)  # assuming per epoch evaluation

    # Loss and optimization
    loss_fn: str = Field(default="torch.nn.CrossEntropyLoss")
    optim: str = Field(default="sgd")
    momentum: float = Field(default=0.9)
    weight_decay: float = Field(default=1e-4)

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="alexnet-training")
    wandb_run_name: str = Field(default="alexnet-training")
    wandb_tags: list[str] = Field(default_factory=lambda: ["tt-xla", "model:torch", "plugin", "wandb"])
    wandb_watch_mode: str = Field(default="gradients")
    wandb_log_freq: int = Field(default=100)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=100)
    epoch_freq: int = Field(default=1)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")
    checkpoint_path: str = Field(default="")
    checkpoint_metric: str = Field(default="val/loss")
    checkpoint_metric_mode: str = Field(default="min")
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=1, ge=0)
    save_strategy: str = Field(default="epoch")
    project_dir: str = Field(default="experiments/results/alexnet")
    save_optim: bool = Field(default=False)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=42)
    deterministic: bool = Field(default=False)

    # Multi-chip settings
    parallelism: str = Field(default="single")  # AlexNet defaults to single-chip
    mesh_shape: str = Field(default="1,1")  # not used for single

    # Other settings
    device: str = Field(default="TT")
    experiment_name: str = Field(default="alexnet-training")
    output_dir: str = Field(default="experiments/results/alexnet")
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)
