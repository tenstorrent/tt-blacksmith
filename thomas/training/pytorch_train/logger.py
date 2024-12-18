# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import pathlib

import torch
import wandb


class PyTorchWandbLogger:
    def __init__(self, config):
        self.config = config.logger_config
        self.project_name = config.experiment_name

        self.run = wandb.init(project=self.project_name)
        self.run.config.update(config.model.dict())
        self.run.config.update(config.training_config.dict())
        self.run.config.batch_size = config.data_loading.batch_size
        self.run.config.max_length = config.data_loading.max_length
        self.run.config.dataset_id = config.data_loading.dataset_id

        self.log_dir = os.path.join(self.config.wandb_dir, self.project_name, self.run.id)
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def log_metrics(self, metrics: dict, step: int = None):
        if step:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def save_checkpoint_step(self, model, optimizer, global_step):
        if self.config.log_every_n_steps is None:
            return
        if global_step % self.config.log_every_n_steps != 0:
            return

        name = f"checkpoint_step_{global_step}.pt"
        filename = os.path.join(self.checkpoints_dir, name)

        checkpoint = {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)
        self._add_artifact(name, filename)
        print(f"Checkpoint saved: {filename}")

    def save_checkpoint_epoch(self, model, optimizer, epoch, loss, validation_loss):
        if self.config.log_every_n_epochs is None:
            return
        if epoch % self.config.log_every_n_epochs != 0:
            return

        name = f"checkpoint_epoch_{epoch}.pt"
        filename = os.path.join(self.checkpoints_dir, name)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "validation_loss": validation_loss,
        }

        torch.save(checkpoint, filename)
        self._add_artifact(name, filename)
        print(f"Checkpoint saved: {filename}")

    def _add_artifact(self, name, filename):
        artifact = wandb.Artifact(name=name, type="model")
        path = pathlib.Path(filename).resolve()
        artifact.add_reference(f"file://{path}")
        self.run.log_artifact(artifact)

    def watch_model(self, model, log="all", log_freq=10):
        wandb.watch(model, log=log, log_freq=log_freq)
