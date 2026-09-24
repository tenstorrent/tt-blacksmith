# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field


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
