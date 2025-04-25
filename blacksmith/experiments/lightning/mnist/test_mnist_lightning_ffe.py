# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from blacksmith.tools.forge_tooling import disable_forge_logger
from blacksmith.tools.torch_lightning import (
    TTLightningModel,
    GradientCheckpoint,
    TTWandbLogger,
    SaveCheckpointArtifact,
)
from blacksmith.tools.cli import generate_config
from blacksmith.datasets.mnist.dataloader import load_mnist_torch
from blacksmith.models.torch.mnist.mnist_linear import MNISTLinear
from blacksmith.experiments.lightning.mnist.configs import ExperimentConfig

import forge
import torch


def test_training():
    # Currently, forge prints a log on every call of forward and backward, disabling it for now
    disable_forge_logger()

    config: ExperimentConfig = generate_config(
        ExperimentConfig, "blacksmith/experiments/lightning/mnist/test_mnist_lightning_ffe.yaml"
    )
    logger_config = config.logger_config

    train_loader, test_loader = load_mnist_torch(
        dtype=eval(config.data_loading_config.dtype), batch_size=config.data_loading_config.batch_size
    )
    model = MNISTLinear(**config.net_config.model_dump())
    logger = TTWandbLogger(
        project=config.experiment_name,
        tags=config.tags,
        save_dir=logger_config.wandb_dir,
    )
    if logger_config.log_hyperparameters:
        logger.experiment.config.update(config.model_dump())

    L_model = TTLightningModel(
        model=model,
        loss=eval(config.loss),
        logger_config=logger_config,
        input_size=config.net_config.input_size,
        output_size=config.net_config.output_size,
        batch_size=config.data_loading_config.batch_size,
        lr=config.training_config.lr,
    )

    callbacks = []
    checkpoint_config = logger_config.checkpoint
    if checkpoint_config.log_checkpoint:
        # Callback for saving checkpoints every n global steps
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_config.checkpoint_dir,
                every_n_train_steps=checkpoint_config.log_every_n_steps,
                filename=f"{logger.experiment.name}/{checkpoint_config.checkpoint_name}",
                save_top_k=checkpoint_config.save_top_k,
            )
        )
        # Callback to send the artifact to wandb with references to the checkpoints of the one epoch
        callbacks.append(SaveCheckpointArtifact())
    if checkpoint_config.save_gradients:
        # Callback for saving gradients inside checkpoint
        callbacks.append(GradientCheckpoint())

    trainer = L.Trainer(max_epochs=config.training_config.epochs, logger=logger, callbacks=callbacks)
    trainer.fit(L_model, train_loader, test_loader)


if __name__ == "__main__":
    test_training()
