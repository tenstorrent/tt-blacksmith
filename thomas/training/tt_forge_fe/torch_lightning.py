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
from thomas.training.logger_config import LoggerConfig
from thomas.tooling.wandb_utils import log_histogram


class TTLightningConfig(BaseModel):
    batch_size: int
    input_size: int
    loss: TorchLoss
    lr: float


class TTLightningModel(L.LightningModule):
    """
    Wrapper around the model to use it with PyTorch Lightning for forge-fe models
    """

    def __init__(self, config: TTLightningConfig, model: nn.Module, logger_config: LoggerConfig):
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
        self.logger_config = logger_config

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
        if self.logger_config.log_train_loss:
            self.log(self.logger_config.log_train_loss, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        acc = (pred.argmax(1) == y).type(torch.float).mean()
        if self.logger_config.log_val_loss:
            self.log(self.logger_config.log_val_loss, loss)
        if self.logger_config.log_val_accuracy:
            self.log(self.logger_config.log_val_accuracy, acc)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.config.lr)

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
        print("Epoch start", self.current_epoch)
        if self.logger_config.log_weights is None:
            return
        if self.logger_config.log_every_n_epochs is None:
            return
        if self.current_epoch % self.logger_config.log_every_n_epochs != 0:
            return
        print("Logging weights", self.current_epoch)
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
            print("Saving Logging checkpoint", trainer.current_epoch)
            trainer.logger.log_checkpoints()
