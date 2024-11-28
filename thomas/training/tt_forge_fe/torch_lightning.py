# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.utilities import _scan_checkpoints
import torch
from pydantic import BaseModel
from torch import nn
import wandb

from thomas.models.torch.loss import TorchLoss
from thomas.training.logging_config import LoggerConfig
from thomas.tooling.wandb_utils import log_histogram


class LightningConfig(BaseModel):
    batch_size: int
    input_size: int
    loss: TorchLoss
    lr: float


class TTLightningModel(L.LightningModule):
    """
    Wrapper around the model to use it with PyTorch Lightning for forge-fe models
    """

    def __init__(self, config: LightningConfig, model: nn.Module, logging_config: LoggerConfig):
        super(TTLightningModel, self).__init__()
        import forge
        # self.save_hyperparameters(config.model_dump())
        self.framework_model = model
        tt_model = forge.compile(
            self.framework_model,
            sample_inputs=[torch.rand(config.batch_size, config.input_size)],
            loss=config.loss(),
        )
        self.config = config
        self.model = tt_model
        self.loss = config.loss()
        self.logging_config = logging_config

    def forward(self, x):
        logits = self.model(x)
        logits = logits[0]
        return logits

    def backward(self, loss, *args, **kwargs):
        loss.backward(*args, **kwargs)
        self.model.backward()

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        if self.logging_config.log_train_loss:
            self.log(self.logging_config.log_train_loss, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        acc = (pred.argmax(1) == y).type(torch.float).mean()
        if self.logging_config.log_val_loss:
            self.log(self.logging_config.log_val_loss, loss)
        if self.logging_config.log_val_accuracy:
            self.log(self.logging_config.log_val_accuracy, acc)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.config.lr)

    def on_after_backward(self):
        print()
        if self.logging_config.log_gradients is None and self.logging_config.log_weights is None:
            return
        if self.global_step % self.logging_config.log_every_n_steps != 0:
            return
        for name, param in self.framework_model.named_parameters():
            if param.grad is None:
                continue
            if self.logging_config.log_gradients:
                log_histogram(
                    self.logger.experiment,
                    self.logging_config.log_gradients.format(name=name),
                    param.grad,
                    self.global_step,
                )
            if self.logging_config.log_weights:
                log_histogram(
                    self.logger.experiment, self.logging_config.log_weights.format(name=name), param, self.global_step
                )


class GradientCheckpoint(L.Callback):
    """
    Callback to save the gradients of the model in the checkpoint
    """

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["gradients"] = {
            name: param.grad for name, param in pl_module.named_parameters() if param.grad is not None
        }


class CustomLogger(WandbLogger):
    """
    Custom logger to log the model checkpoints as artifacts, and everything else as usual
    """

    def __init__(self, *args, **kwargs):
        WandbLogger.__init__(self, *args, **kwargs)
        self.logged_model_time = {}
        self.checkpoint_artifact = None

    def after_save_checkpoint(self, checkpoint_callback):
        # Get all checkpoint that are not logged yet
        models_to_log = _scan_checkpoints(checkpoint_callback, self.logged_model_time)

        # Add only new references
        for save_time, model_path, _, tag in models_to_log:
            if model_path not in self.logged_model_time:
                self.checkpoint_artifact.add_reference(f"file://{model_path}")

        # Update the logged_model_time, in case the model is saved again, or new checkpoints are added
        self.logged_model_time.update({model_path: save_time for save_time, model_path, _, _ in models_to_log})

    def log_checkpoints(self):
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


class SaveChecpointArtifact(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if isinstance(trainer.logger, CustomLogger):
            trainer.logger.create_artifact()

    def on_train_epoch_end(self, trainer, pl_module):
        if isinstance(trainer.logger, CustomLogger):
            trainer.logger.log_checkpoints()
