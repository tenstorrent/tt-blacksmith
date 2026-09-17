# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import traceback

from blacksmith.tools.trainer.callback import Callback


class MetricsCallback(Callback):
    """
    Default metrics-logging callback for the `Trainer`.
    """

    def __init__(self):
        # Sum of per-step average losses since the last log; divided by
        # `steps_freq` (exactly that many steps complete between logs).
        self._running_loss = 0.0
        self._prev_global_step = 0

    def on_train_start(self, trainer, *args, **kwargs):
        # `on_train_start` fires before the trainer validates its config, so guard.
        if trainer.config is None or trainer.logger is None:
            return

        self._prev_global_step = trainer.global_step

        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        model_info = {
            "model_name": trainer.config.model_name,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "train_dataset_size": self._dataloader_size(trainer.train_dataloader, trainer.config.batch_size),
            "val_dataset_size": self._dataloader_size(trainer.val_dataloader, trainer.config.batch_size),
        }
        trainer.logger.log_model_info(model_info)

        if trainer.config.logging.model_to_wandb:
            trainer.logger.watch_model(trainer.model)

    def on_optimizer_step_end(self, trainer, window_loss, *args, **kwargs):
        # optimizer_step has already synced; window_loss is the sum of scaled
        # micro-batch losses for this accumulation window (the per-step mean).
        self._running_loss += window_loss.item()

    def on_train_batch_end(self, trainer, *args, **kwargs):
        if trainer.logger is None:
            return

        # Only finalize when an optimizer step completed this batch. Under
        # gradient accumulation most micro-batches do not advance global_step.
        if trainer.global_step == self._prev_global_step:
            return
        self._prev_global_step = trainer.global_step

        steps_freq = trainer.config.metrics.steps_freq
        if trainer.global_step % steps_freq == 0:
            available = {"loss": self._running_loss / steps_freq}
            metrics = self._select(available, trainer.config.metrics.train_metrics, phase="train")
            if metrics:
                trainer.logger.log_metrics(metrics, commit=False, step=trainer.global_step)
            self._running_loss = 0.0

        # Flush the batched W&B logs so train and validation land on the same step.
        trainer.logger.log_metrics({}, commit=True, step=trainer.global_step)

    def on_validation_end(self, trainer, val_loss, *args, **kwargs):
        if trainer.logger is None:
            return
        available = {"loss": val_loss}
        metrics = self._select(available, trainer.config.metrics.val_metrics, phase="val")
        if metrics:
            trainer.logger.log_metrics(metrics, commit=False, step=trainer.global_step)

    def on_error(self, trainer, exception, *args, **kwargs):
        if trainer.logger is not None:
            trainer.logger.error(f"Training failed with error: {exception}", traceback.format_exc())

    @staticmethod
    def _select(available: dict[str, float], requested: list[str], phase: str) -> dict[str, float]:
        # Keep only requested metrics the trainer can currently provide; unknown
        # names are ignored so the metric set stays forward-compatible.
        return {f"{phase}/{name}": available[name] for name in requested if name in available}

    @staticmethod
    def _dataloader_size(dataloader, batch_size):
        if dataloader is None:
            return None
        try:
            return len(dataloader) * batch_size
        except TypeError:
            return None
