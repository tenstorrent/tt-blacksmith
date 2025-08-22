# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field
from typing import Tuple


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_name: str = Field(default="tiny-imagenet")
    data_path: str = Field(default="tiny-imagenet-200")
    
    # Model settings
    model_variant: str = Field(default="alexnet")
    num_classes: int = Field(default=200, gt=0)
    image_size: int = Field(default=64, gt=0)
    dtype: str = Field(default="torch.float32")
    
    # Training hyperparameters
    learning_rate: float = Field(default=0.01, gt=0)
    batch_size: int = Field(default=256, gt=0)
    momentum: float = Field(default=0.9, ge=0, le=1)
    weight_decay: float = Field(default=1e-4, ge=0)
    num_epochs: int = Field(default=90, gt=0)
    optim: str = Field(default="sgd")
    
    # Learning rate scheduler
    lr_scheduler: str = Field(default="step")
    lr_step_size: int = Field(default=30, gt=0)
    lr_gamma: float = Field(default=0.1, gt=0, le=1)
    
    # Data loading
    num_workers: int = Field(default=8, ge=0)
    eval_split: float = Field(default=0.1, gt=0, lt=1)
    
    # Other settings
    seed: int = Field(default=42)
    output_dir: str = Field(default="experiments/results/alexnet")
    report_to: str = Field(default="wandb")
    wandb_project: str = Field(default="alexnet-training")
    wandb_watch_mode: str = Field(default="gradients")
    wandb_log_freq: int = Field(default=100)
    save_strategy: str = Field(default="epoch")
    logging_strategy: str = Field(default="steps")
    logging_steps: int = Field(default=100, gt=0)
    save_total_limit: int = Field(default=3, gt=0)
    do_train: bool = Field(default=True)
    do_eval: bool = Field(default=True) 