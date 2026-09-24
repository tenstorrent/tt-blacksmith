# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field


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
