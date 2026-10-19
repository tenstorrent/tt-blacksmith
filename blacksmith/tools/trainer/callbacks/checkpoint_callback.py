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

    def on_train_start(self, trainer, *args, **kwargs):
        # The trainer builds a framework-specific manager so this callback stays
        # free of framework imports.
        self.manager = trainer.create_checkpoint_manager(trainer.config.checkpoint, trainer.logger)

        if trainer.config.checkpoint.init_from_checkpoint:
            # Model and optimizer state only: training still starts at step 0,
            # so the returned step / epoch / metrics are unused.
            self.manager.load_checkpoint(trainer.model, trainer.optimizer)

    def on_validation_end(self, trainer, val_loss, *args, **kwargs):
        # Cache the freshest validation loss so saved checkpoints carry it as a
        # metric (used for best-N tracking). Runs before that step's `on_step_end`.
        # Read the device-side loss back to a host float once, here (off the hot
        # path): it is serialized to JSON and compared for best-N selection.
        self._last_metrics["val/loss"] = trainer.loss_to_float(val_loss)

    def on_step_end(self, trainer, global_step, *args, **kwargs):
        if self.manager.should_save_checkpoint(global_step):
            self.manager.save_checkpoint(
                trainer.model,
                step=global_step,
                epoch=trainer.epoch,
                optimizer=trainer.optimizer,
                metrics=dict(self._last_metrics),
            )

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        if self.manager.should_save_checkpoint(trainer.global_step, epoch=trainer.epoch):
            self.manager.save_checkpoint(
                trainer.model,
                step=trainer.global_step,
                epoch=trainer.epoch,
                optimizer=trainer.optimizer,
                metrics=dict(self._last_metrics),
            )

    def on_train_end(self, trainer, *args, **kwargs):
        if trainer.config.checkpoint.save_strategy == "none":
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
        trainer.logger.log_artifact(final_path, artifact_type="model", name=final_name)
