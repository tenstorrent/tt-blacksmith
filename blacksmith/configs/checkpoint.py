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
    # At least 1, so the checkpoint just saved (e.g. the final one) is never rotated away.
    keep_last_n: int = Field(ge=1)
    keep_best_n: int = Field(ge=0)
    # Disables rotation: no checkpoint file is ever deleted and every save stays in the history.
    keep_all_checkpoints: bool = False
    checkpoint_metric: str
    checkpoint_metric_mode: str  # [min, max]

    # Initialize model / optimizer state from a checkpoint. Not a resume: training still starts at step 0.
    init_from_checkpoint: bool
    init_option: str  # [last, best, path]
    checkpoint_path: str  # path to checkpoint if init_option is "path"
