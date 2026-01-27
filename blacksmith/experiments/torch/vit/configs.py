# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import BaseModel, Field

from blacksmith.tools.test_config import TestConfig


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="stanfordcars")
    image_size: int = Field(default=224, gt=0)
    image_mean: list[float] = Field(default=[0.5, 0.5, 0.5])
    image_std: list[float] = Field(default=[0.5, 0.5, 0.5])

    # Model settings
    model_name: str = Field(default="google/vit-base-patch16-224")
    dtype: str = Field(default="torch.bfloat16")
    ignored_index: int = Field(default=-100)

    # Training hyperparameters
    learning_rate: float = Field(default=1e-3, gt=0)
    batch_size: int = Field(default=10, gt=0)
    num_epochs: int = Field(default=8, gt=0)

    # Loss
    loss_fn: str = Field(default="torch.nn.CrossEntropyLoss")

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="vit-finetuning")
    wandb_run_name: str = Field(default="tt-vit-stanfordcars")
    wandb_tags: list[str] = Field(default_factory=lambda: ["test"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=10)
    epoch_freq: int = Field(default=1)
    val_steps_freq: int = Field(default=50)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")  # [last, best, path]
    checkpoint_path: str = Field(default="")  # path to checkpoint if resume_option is "path"
    checkpoint_metric: str = Field(default="eval/loss")
    checkpoint_metric_mode: str = Field(default="min")  # [min, max]
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=3, ge=0)
    save_strategy: str = Field(default="epoch")
    project_dir: str = Field(default="blacksmith/experiments/torch/vit")
    save_optim: bool = Field(default=False)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=23)
    deterministic: bool = Field(default=False)

    # Device settings
    parallelism_strategy: str = Field(default="single")  # [single, data_parallel, tensor_parallel]
    mesh_shape: str = Field(default="1,2")  # Used if parallelism_strategy != single
    tp_sharding_specs: dict[str, list[Optional[int]]] = Field(default_factory=dict)  # Used for model tp sharding

    # LoRA setup
    lora_r: int = Field(default=4, gt=0)
    lora_alpha: int = Field(default=8, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])
    lora_dropout: float = Field(default=0.1, ge=0, le=1)

    # Other settings
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)

    # Test settings
    test_config: Optional[TestConfig] = Field(default=None)
