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

    log_level: str
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str
    wandb_tags: list[str]
    wandb_watch_mode: str
    wandb_log_freq: int
    model_to_wandb: bool
