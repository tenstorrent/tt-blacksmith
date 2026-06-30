# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import traceback

from blacksmith.tools.trainer.callback import Callback


class MetricsCallback(Callback):
    """
    Default metrics-logging callback for the ``Trainer``.

    Uses the trainer-owned logger (``trainer.logger``) to report hyperparameters,
    model/dataset info, and the metrics requested in ``trainer.config.metrics``
    (a ``MetricsConfig``). The logger lifecycle (creation / ``finish``) is owned by
    the ``Trainer``; this callback only decides what to log and when.

    Which metrics are logged is driven by ``MetricsConfig.train_metrics`` /
    ``val_metrics`` and the cadence by ``MetricsConfig.steps_freq``. Logger setup
    (W&B, log level) lives separately in ``LoggingConfig``.
    """

    def __init__(self):
        self._running_loss = 0.0
        self._last_loss = 0.0
        self._prev_global_step = 0

    def on_train_start(self, trainer):
        # ``on_train_start`` fires before the trainer validates its config, so guard.
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

    def on_forward_end(self, trainer, loss):
        # Track the latest micro-batch loss; it becomes the loss for the
        # optimizer step that this micro-batch contributes to.
        self._last_loss = loss.item()

    def on_train_batch_end(self, trainer):
        if trainer.logger is None:
            return

        # Only act when an optimizer step completed this batch. Under gradient
        # accumulation most batches do not advance ``global_step``.
        if trainer.global_step == self._prev_global_step:
            return
        self._prev_global_step = trainer.global_step

        # Accumulate one loss per optimizer step. Between two logs exactly
        # ``steps_freq`` optimizer steps complete, so the running sum always
        # holds ``steps_freq`` values when we log.
        self._running_loss += self._last_loss

        steps_freq = trainer.config.metrics.steps_freq
        if trainer.global_step % steps_freq == 0:
            available = {"loss": self._running_loss / steps_freq}
            metrics = self._select(available, trainer.config.metrics.train_metrics, phase="train")
            if metrics:
                trainer.logger.log_metrics(metrics, commit=False, step=trainer.global_step)
            self._running_loss = 0.0

        # Flush the batched W&B logs so train and validation land on the same step.
        trainer.logger.log_metrics({}, commit=True, step=trainer.global_step)

    def on_validation_end(self, trainer, val_loss):
        if trainer.logger is None:
            return
        available = {"loss": val_loss}
        metrics = self._select(available, trainer.config.metrics.val_metrics, phase="val")
        if metrics:
            trainer.logger.log_metrics(metrics, commit=False, step=trainer.global_step)

    def on_error(self, trainer, exception):
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
