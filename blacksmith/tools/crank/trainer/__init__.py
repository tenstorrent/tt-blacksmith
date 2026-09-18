# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-crank Trainer pipeline.

Same shape as `blacksmith.tools.trainer`: `Trainer` runs the loop, strategies fill in
model / data / loss, callbacks do metrics and checkpointing. `Callback`,
`CallbackHandler` and `MetricsCallback` are backend-agnostic and re-exported from the
tt-xla package; `Trainer` and `CheckpointCallback` are the tt-crank versions.
"""
from blacksmith.tools.crank.trainer.callbacks import CheckpointCallback
from blacksmith.tools.crank.trainer.trainer import Trainer
from blacksmith.tools.trainer.callback import Callback
from blacksmith.tools.trainer.callbacks import MetricsCallback
from blacksmith.tools.trainer.callbacks_handler import CallbackHandler

__all__ = [
    "Trainer",
    "Callback",
    "CallbackHandler",
    "MetricsCallback",
    "CheckpointCallback",
]
