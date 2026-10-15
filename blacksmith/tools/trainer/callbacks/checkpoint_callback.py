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
        self._last_metrics: dict[str, float] = {}
        self._prev_global_step = 0

    def on_train_start(self, trainer, *args, **kwargs):
        # `on_train_start` fires before the trainer validates its config, so guard.
        if trainer.config is None:
            return

        # The trainer builds a framework-specific manager so this callback stays
        # free of framework imports.
        self.manager = trainer.create_checkpoint_manager(trainer.config.checkpoint, trainer.logger)
        self._prev_global_step = trainer.global_step

        if trainer.config.checkpoint.resume_from_checkpoint:
            self.manager.load_checkpoint(trainer.model, trainer.optimizer)

    def on_validation_end(self, trainer, val_loss, *args, **kwargs):
        # Cache the freshest validation loss so saved checkpoints carry it as a
        # metric (used for best-N tracking). Runs before `on_train_batch_end`.
        # Read the device-side loss back to a host float once, here (off the hot
        # path): it is serialized to JSON and compared for best-N selection.
        self._last_metrics["val/loss"] = trainer.loss_to_float(val_loss)

    def on_train_batch_end(self, trainer, *args, **kwargs):
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
                metrics=dict(self._last_metrics),
            )

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        if self.manager is None:
            return

        if self.manager.should_save_checkpoint(trainer.global_step, epoch=trainer.epoch):
            self.manager.save_checkpoint(
                trainer.model,
                step=trainer.global_step,
                epoch=trainer.epoch,
                optimizer=trainer.optimizer,
                metrics=dict(self._last_metrics),
            )

    def on_train_end(self, trainer, *args, **kwargs):
        if self.manager is None:
            return

        final_name = trainer.config.checkpoint.final_checkpoint_name
        final_path = self.manager.save_checkpoint(
            trainer.model,
            step=trainer.global_step,
            epoch=trainer.epoch,
            optimizer=trainer.optimizer,
            metrics=dict(self._last_metrics),
            checkpoint_name=final_name,
        )
        if trainer.logger is not None:
            trainer.logger.log_artifact(final_path, artifact_type="model", name=final_name)
