# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import BaseModel, Field

from blacksmith.tools.test_config import TestConfig


class TrainingConfig(BaseModel):
    # Dataset settings
    dataset_id: str = Field(default="wikitext")

    # Model settings
    model_name: str = Field(default="openai/gpt-oss-20b")
    max_length: int = Field(default=512, gt=0)
    dtype: str = Field(default="torch.bfloat16")

    # Training hyperparameters
    training_type: str = Field(default="lora")
    learning_rate: float = Field(default=3e-4, gt=0)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    gradient_checkpointing: bool = Field(default=True)
    weight_decay: float = Field(default=0.1, ge=0)
    num_epochs: int = Field(default=1, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    optim: str = Field(default="adamw_torch")
    ignored_index: int = Field(default=-100)

    # Logging settings
    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=False)
    wandb_project: str = Field(default="gpt-oss-20b-ep")
    wandb_run_name: str = Field(default="gpt-oss-ep-run")
    wandb_tags: list[str] = Field(default_factory=lambda: ["expert-parallel", "lora", "gpt-oss"])
    wandb_watch_mode: str = Field(default="gradients")
    wandb_log_freq: int = Field(default=100)
    model_to_wandb: bool = Field(default=False)
    steps_freq: int = Field(default=10)
    val_steps_freq: int = Field(default=100)
    epoch_freq: int = Field(default=1)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")  # [last, best, path]
    checkpoint_path: str = Field(default="")  # path to checkpoint if resume_option is "path"
    checkpoint_metric: str = Field(default="val/loss")
    checkpoint_metric_mode: str = Field(default="min")  # [min, max]
    keep_last_n: int = Field(default=2, ge=0)
    keep_best_n: int = Field(default=1, ge=0)
    save_strategy: str = Field(default="step")
    project_dir: str = Field(default="blacksmith/experiments/torch/gpt_oss/distributed")
    save_optim: bool = Field(default=False)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=42)
    deterministic: bool = Field(default=False)

    # LoRA config
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0.0)
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    lora_task_type: str = Field(default="CAUSAL_LM")

    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=False)

    test_config: Optional[TestConfig] = Field(default=None)
