# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-crank Trainer pipeline.

Standalone copy of `blacksmith.tools.trainer` (the tt-xla pipeline) with the tt-xla
lazy-graph machinery removed: `Trainer` runs the loop, strategies fill in model / data /
loss, callbacks do metrics and checkpointing. Nothing here imports from
`blacksmith.tools.trainer`.
"""
from blacksmith.tools.crank.trainer.callback import Callback
from blacksmith.tools.crank.trainer.callbacks import CheckpointCallback, MetricsCallback
from blacksmith.tools.crank.trainer.callbacks_handler import CallbackHandler
from blacksmith.tools.crank.trainer.trainer import Trainer

__all__ = [
    "Trainer",
    "Callback",
    "CallbackHandler",
    "MetricsCallback",
    "CheckpointCallback",
]
