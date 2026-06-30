# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """
    Logger / Weights & Biases setup consumed by ``TrainingLogger``.

    Holds only logger configuration; *what* metrics get logged and *how often*
    lives in :class:`MetricsConfig`. Designed to be composed as a nested
    sub-config (e.g. ``TrainerConfig.logging``).
    """

    log_level: str = Field(default="INFO")
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="model-finetuning")
    wandb_run_name: str = Field(default="tt-model")
    wandb_tags: list[str] = Field(default_factory=lambda: ["test"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)


class MetricsConfig(BaseModel):
    """
    Declares which metrics to log and how often.

    Separated from :class:`LoggingConfig` (logger/W&B setup) so trainings can
    extend the set of logged metrics without touching logger configuration.
    Designed to be composed as a nested sub-config (e.g. ``TrainerConfig.metrics``).
    """

    # Cadence at which train/validation metrics are logged.
    steps_freq: int = Field(default=25, ge=1)
    epoch_freq: int = Field(default=1, ge=1)

    # Metric names to log, per phase. The callback logs each name it can resolve
    # (currently "loss"); names without a source yet are ignored, keeping the set
    # forward-compatible with metrics future trainers expose.
    train_metrics: list[str] = Field(default_factory=lambda: ["loss"])
    val_metrics: list[str] = Field(default_factory=lambda: ["loss"])


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
    final_checkpoint_name: str = Field(default="final_model.pth")
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
