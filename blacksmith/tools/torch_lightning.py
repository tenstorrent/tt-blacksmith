# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import forge
import lightning as L
import numpy as np
import torch
import wandb
from forge.tensor import to_forge_tensors
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from torch import nn

from blacksmith.tools.logging.configs import LoggerConfig


def log_histogram(experiment, name, tensor, step, n_bins=100):
    """
    Log a histogram of the tensor to wandb.
    Currently it supports numpy and torch tensors.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach().numpy()

    if tensor.ndim != 1:
        tensor = tensor.flatten()

    hist, bins = np.histogram(tensor, bins=n_bins)

    experiment._log(
        {
            name: wandb.Histogram(np_histogram=(hist, bins)),
        },
        step=step,
    )


class TTLightningModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss,
        logger_config: LoggerConfig,
        input_size: int,
        output_size: int,
        batch_size: int,
        lr: float,
    ):
        super(TTLightningModel, self).__init__()

        self.framework_model = model
        self.logger_config = logger_config
        self.lr = lr

        tt_model = forge.compile(
            self.framework_model,
            sample_inputs=[torch.rand(batch_size, input_size)],
            training=True,
        )

        self.model = tt_model
        loss_inputs = [
            torch.rand(batch_size, output_size).requires_grad_(True),
            torch.rand(batch_size, output_size),
        ]
        loss_inputs = to_forge_tensors(loss_inputs)
        self.loss_on_cpu = issubclass(loss, torch.nn.modules.loss._Loss)
        if self.loss_on_cpu:
            self.loss_module = loss()
        else:
            self.loss_module = loss(type(loss).__name__)
            self.loss_module = forge.compile(
                self.loss_module, sample_inputs=loss_inputs, attach_to=tt_model, training=True
            )

    def forward(self, x):
        logits = self.model(x)
        logits = logits[0]
        return logits

    def backward(self, loss, *args, **kwargs):
        if self.loss_on_cpu:
            loss.backward()
            self.model.backward()
        else:
            self.loss_module.backward()

    def calculate_loss(self, pred, target):
        loss = self.loss_module(pred, target)
        return loss if self.loss_on_cpu else loss[0]

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.calculate_loss(pred, y)
        if self.logger_config.log_train_loss:
            self.log(self.logger_config.log_train_loss, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.calculate_loss(pred, y)
        acc = (pred.argmax(1) == y.argmax(1)).type(torch.float).mean()
        if self.logger_config.log_val_loss:
            self.log(self.logger_config.log_val_loss, loss)
        if self.logger_config.log_val_accuracy:
            self.log(self.logger_config.log_val_accuracy, acc)

    def configure_optimizers(self):
        return torch.optim.SGD(self.framework_model.parameters(), lr=self.lr)

    def on_after_backward(self):
        if self.logger_config.log_gradients is None:
            return
        if self.logger_config.log_every_n_steps is None:
            return
        if self.global_step % self.logger_config.log_every_n_steps != 0:
            return
        for name, param in self.framework_model.named_parameters():
            if param.grad is None:
                continue
            log_histogram(
                self.logger.experiment,
                self.logger_config.log_gradients.format(name=name),
                param.grad,
                self.global_step,
            )

    def on_train_batch_start(self, batch, batch_idx):
        if self.logger_config.log_weights is None:
            return
        if self.logger_config.log_every_n_steps is None:
            return
        if self.global_step % self.logger_config.log_every_n_steps != 0:
            return
        for name, param in self.framework_model.named_parameters():
            log_histogram(
                self.logger.experiment,
                self.logger_config.log_weights.format(name=name),
                param,
                self.global_step,
            )

    def on_train_epoch_start(self):
        if self.logger_config.log_weights is None:
            return
        if self.logger_config.log_every_n_epochs is None:
            return
        if self.current_epoch % self.logger_config.log_every_n_epochs != 0:
            return
        for name, param in self.framework_model.named_parameters():
            log_histogram(
                self.logger.experiment,
                self.logger_config.log_weights.format(name=name),
                param,
                self.global_step,
            )


class GradientCheckpoint(L.Callback):
    """
    Callback to save the gradients of the model in the checkpoint
    """

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["gradients"] = {
            name: param.grad for name, param in pl_module.named_parameters() if param.grad is not None
        }


class TTWandbLogger(WandbLogger):
    """
    Custom logger to log the model checkpoints as artifacts, and everything else as usual
    """

    def __init__(self, *args, **kwargs):
        WandbLogger.__init__(self, *args, **kwargs)
        # Key is path to the checkpoint, value is the timestamp when the checkpoint was saved
        # In case the model is saved again, the timestamp is updated, and the checkpoint is added to the artifact
        self.checkpoint_save_timestamp = {}
        self.checkpoint_artifact = None

    def after_save_checkpoint(self, checkpoint_callback):
        # Get all checkpoints that are not logged yet
        models_to_log = _scan_checkpoints(checkpoint_callback, self.checkpoint_save_timestamp)

        if self.checkpoint_artifact is None:
            self.create_artifact()

        # Add only new checkpoints to the artifact
        for save_time, model_path, _, tag in models_to_log:
            if model_path not in self.checkpoint_save_timestamp:
                self.checkpoint_artifact.add_reference(f"file://{model_path}")

        # Update the checkpoint_save_timestamp, in case the model is saved again, or new checkpoints are added
        self.checkpoint_save_timestamp.update({model_path: save_time for save_time, model_path, _, _ in models_to_log})

    def log_checkpoints(self):
        """
        Log the checkpoints as artifacts, after that no new references can be added
        """
        if self.checkpoint_artifact is not None:
            self.experiment.log_artifact(self.checkpoint_artifact)
            self.checkpoint_artifact = None

    def create_artifact(self):
        artifact_name = f"model-{self.experiment.name}"
        self.checkpoint_artifact = wandb.Artifact(artifact_name, type="model")

    def finalize(self, status):
        if self.checkpoint_artifact is not None:
            self.log_checkpoints()
        super().finalize(status)


class SaveCheckpointArtifact(L.Callback):
    """
    Save the model checkpoints as artifacts at the end of the epoch
    """

    def on_train_epoch_end(self, trainer, pl_module):
        if isinstance(trainer.logger, TTWandbLogger):
            trainer.logger.log_checkpoints()
