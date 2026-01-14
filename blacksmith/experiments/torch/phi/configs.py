# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="sst2")

    # Model settings
    model_name: str = Field(default="microsoft/phi-1")
    max_length: int = Field(default=32, gt=0)
    dtype: str = Field(default="torch.bfloat16")
    ignored_index: int = Field(default=-100)

    # Training hyperparameters
    training_type: str = Field(default="lora")  # [lora, adapters]
    learning_rate: float = Field(default=6e-5, gt=0)
    batch_size: int = Field(default=8, gt=0)
    gradient_checkpointing: bool = Field(default=False)
    num_epochs: int = Field(default=1, gt=0)
    optim: str = Field(default="adamw_torch")

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="phi1-finetuning")
    wandb_run_name: str = Field(default="tt-phi1-sst2")
    wandb_tags: list[str] = Field(default_factory=lambda: ["test"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=10)
    epoch_freq: int = Field(default=1)
    val_steps_freq: int = Field(default=50)
    print_examples: bool = Field(default=True)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")  # [last, best, path]
    checkpoint_path: str = Field(default="")  # path to checkpoint if resume_option is "path"
    checkpoint_metric: str = Field(default="eval/loss")
    checkpoint_metric_mode: str = Field(default="min")  # [min, max]
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=3, ge=0)
    save_strategy: str = Field(default="epoch")
    project_dir: str = Field(default="blacksmith/experiments/torch/phi")
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
    mesh_shape: str = Field(default="8,1")  # Used if parallelism_strategy != single
    tp_sharding_specs: dict[str, list[Optional[int]]] = Field(default_factory=dict)  # Used for model tp sharding

    # LoRA setup
    lora_r: int = Field(default=4, gt=0)
    lora_alpha: int = Field(default=8, gt=0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["all-linear"])
    lora_task_type: str = Field(default="CAUSAL_LM")

    # Other settings
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)
