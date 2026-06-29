# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """
    Reusable logging settings consumed by ``TrainingLogger`` and the logging callback.

    Designed to be composed as a nested sub-config (e.g. ``TrainerConfig.logging``) so
    logging settings live in their own namespace, independent of experiment configs.
    """

    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="model-finetuning")
    wandb_run_name: str = Field(default="tt-model-test")
    wandb_tags: list[str] = Field(default_factory=lambda: ["test"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)

    # Cadence at which train metrics are logged.
    steps_freq: int = Field(default=25, ge=1)
    epoch_freq: int = Field(default=1, ge=1)


class CheckpointConfig(BaseModel):
    """
    Reusable checkpoint settings consumed by ``CheckpointManager`` and the checkpoint callback.

    Designed to be composed as a nested sub-config (e.g. ``TrainerConfig.checkpoint``).
    """

    # Cadence at which checkpoints are saved.
    steps_freq: int = Field(default=25, ge=1)
    epoch_freq: int = Field(default=1, ge=1)
    save_strategy: str = Field(default="epoch")  # [epoch, step, none]

    project_dir: str = Field(default="blacksmith/experiments/torch/model")
    save_optim: bool = Field(default=False)
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=3, ge=0)
    checkpoint_metric: str = Field(default="eval/loss")
    checkpoint_metric_mode: str = Field(default="min")  # [min, max]

    # Storage backend settings.
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Resume settings.
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")  # [last, best, path]
    checkpoint_path: str = Field(default="")  # path to checkpoint if resume_option is "path"
