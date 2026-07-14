# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.trainer.callback import Callback


class CheckpointCallback(Callback):
    """
    Default checkpointing callback for the `Trainer`.
    """

    def __init__(self):
        self.manager: CheckpointManager | None = None
        self._last_val_loss: float | None = None
        self._prev_global_step = 0

    def on_train_start(self, trainer):
        # `on_train_start` fires before the trainer validates its config, so guard.
        if trainer.config is None:
            return

        device = getattr(trainer.device_manager, "device", None)
        self.manager = CheckpointManager(trainer.config.checkpoint, logger=trainer.logger, device=device)
        self._prev_global_step = trainer.global_step

        if trainer.config.checkpoint.resume_from_checkpoint:
            self.manager.load_checkpoint(trainer.model, trainer.optimizer)

    def on_validation_end(self, trainer, val_loss):
        # Cache the freshest validation loss so saved checkpoints carry it as a
        # metric (used for best-N tracking). Runs before `on_train_batch_end`.
        self._last_val_loss = val_loss

    def on_train_batch_end(self, trainer):
        if self.manager is None:
            return

        # Only act when an optimizer step completed this batch. Under gradient
        # accumulation most batches do not advance `global_step`.
        if trainer.global_step == self._prev_global_step:
            return
        self._prev_global_step = trainer.global_step

        if self.manager.should_save_checkpoint(trainer.global_step):
            self.manager.save_checkpoint(
                trainer.model,
                step=trainer.global_step,
                epoch=trainer.epoch,
                optimizer=trainer.optimizer,
                metrics=self._metrics(trainer),
            )

    def on_train_epoch_end(self, trainer):
        if self.manager is None:
            return

        if self.manager.should_save_checkpoint(trainer.global_step, epoch=trainer.epoch):
            self.manager.save_checkpoint(
                trainer.model,
                step=trainer.global_step,
                epoch=trainer.epoch,
                optimizer=trainer.optimizer,
                metrics=self._metrics(trainer),
            )

    def on_train_end(self, trainer):
        if self.manager is None:
            return

        final_name = trainer.config.checkpoint.final_checkpoint_name
        final_path = self.manager.save_checkpoint(
            trainer.model,
            step=trainer.global_step,
            epoch=trainer.epoch,
            optimizer=trainer.optimizer,
            metrics=self._metrics(trainer),
            checkpoint_name=final_name,
        )
        if trainer.logger is not None:
            trainer.logger.log_artifact(final_path, artifact_type="model", name=final_name)

    def _metrics(self, trainer):
        if self._last_val_loss is None:
            return {}
        return {trainer.config.checkpoint.checkpoint_metric: self._last_val_loss}
