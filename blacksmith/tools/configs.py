# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator


class LoggingConfig(BaseModel):
    """
    Logger / Weights & Biases setup consumed by ``TrainingLogger``.

    Holds only logger configuration; *what* metrics get logged and *how often*
    lives in :class:`MetricsConfig`. Designed to be composed as a nested
    sub-config (e.g. ``TrainerConfig.logging``).
    """

    log_level: str
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str
    wandb_tags: list[str]
    wandb_watch_mode: str
    wandb_log_freq: int
    model_to_wandb: bool


class MetricsConfig(BaseModel):
    """
    Declares which metrics to log and how often.

    Separated from :class:`LoggingConfig` (logger/W&B setup) so trainings can
    extend the set of logged metrics without touching logger configuration.
    Designed to be composed as a nested sub-config (e.g. ``TrainerConfig.metrics``).
    """

    # Cadence at which train/validation metrics are logged.
    steps_freq: int = Field(ge=1)
    epoch_freq: int = Field(ge=1)

    # Metric names to log, per phase. The callback logs each name it can resolve
    # (currently "loss"); names without a source yet are ignored, keeping the set
    # forward-compatible with metrics future trainers expose.
    train_metrics: list[str]
    val_metrics: list[str]


class CheckpointConfig(BaseModel):
    """
    Reusable checkpoint settings consumed by ``CheckpointManager`` and the checkpoint callback.

    Designed to be composed as a nested sub-config (e.g. ``TrainerConfig.checkpoint``).
    """

    # Cadence at which checkpoints are saved.
    steps_freq: int = Field(ge=1)
    epoch_freq: int = Field(ge=1)
    save_strategy: str  # [epoch, step, none]

    project_dir: str
    final_checkpoint_name: str
    save_optim: bool
    keep_last_n: int = Field(ge=0)
    keep_best_n: int = Field(ge=0)
    checkpoint_metric: str
    checkpoint_metric_mode: str  # [min, max]

    # Storage backend settings.
    storage_backend: str
    sync_to_storage: bool
    load_from_storage: bool
    remote_path: str

    # Resume settings.
    resume_from_checkpoint: bool
    resume_option: str  # [last, best, path]
    checkpoint_path: str  # path to checkpoint if resume_option is "path"


class CustomDatasetConfig(BaseModel):
    """
    Additional config in case of custom datasets.
    Train and validation sets should be loaded from separate files.
    """

    file_type: str = Field(default="json")
    train_dataset_path: Optional[str] = Field(default=None)
    val_dataset_path: Optional[str] = Field(default=None)

    data_files: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def build_data_files(self) -> "CustomDatasetConfig":
        if self.train_dataset_path is None or self.train_dataset_path == "":
            raise ValueError("train_dataset_path is required and was not provided.")
        self.data_files = {"train": self.train_dataset_path}
        if self.val_dataset_path:
            self.data_files["validation"] = self.val_dataset_path
        return self

    # Define format type (Alpaca-style, chat, etc)
    format: str = Field(default="alpaca")

    column_mapping: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {
            "instruction": "instruction",
            "input": "input",
            "output": "output",
        }
    )
