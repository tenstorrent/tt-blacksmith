# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from blacksmith.tools.crank.checkpoints_manager import CheckpointManager
from blacksmith.tools.trainer.callbacks import (
    CheckpointCallback as XlaCheckpointCallback,
)


class CheckpointCallback(XlaCheckpointCallback):
    """`CheckpointCallback` backed by the tt-crank `CheckpointManager` (host-side, DTensor-gathered saves)."""

    def on_train_start(self, trainer, *args, **kwargs):
        if trainer.config is None:
            return

        device = getattr(trainer.device_manager, "device", None)
        self.manager = CheckpointManager(trainer.config.checkpoint, logger=trainer.logger, device=device)
        self._prev_global_step = trainer.global_step

        if trainer.config.checkpoint.resume_from_checkpoint:
            self.manager.load_checkpoint(trainer.model, trainer.optimizer)
