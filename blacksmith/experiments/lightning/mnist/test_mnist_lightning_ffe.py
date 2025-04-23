# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from blacksmith.tooling.forge_tooling import disable_forge_logger
from blacksmith.training.tt_forge_fe.torch_lightning import (
    TTLightningModel,
    GradientCheckpoint,
    TTWandbLogger,
    SaveCheckpointArtifact,
)
from blacksmith.training.logger_config import LoggerConfig, get_default_logger_config
from blacksmith.models.torch.mnist_linear import MNISTLinear, MNISTLinearConfig
from blacksmith.tooling.cli import generate_config
from blacksmith.tooling.data import load_dataset
from pydantic import BaseModel, Field
from blacksmith.models.config import NetConfig
from blacksmith.models.torch.dtypes import TorchDType
from blacksmith.tooling.config import DataLoadingConfig
from blacksmith.training.config import TrainingConfig


class ExperimentConfig(BaseModel):
    experiment_name: str
    tags: List[str]
    net_config: MNISTLinearConfig
    training_config: TrainingConfig
    data_loading_config: DataLoadingConfig
    logger_config: LoggerConfig = Field(default_factory=get_default_logger_config)


def test_training():
    # Currently, forge prints a log on every call of forward and backward, disabling it for now
    disable_forge_logger()

    config: ExperimentConfig = generate_config(ExperimentConfig, "blacksmith/experiments/test_mnist_lightning_ffe.yaml")
    logger_config = config.logger_config

    train_loader, test_loader = load_dataset(config.data_loading_config)
    model = MNISTLinear(config.net_config)
    logger = TTWandbLogger(
        project=config.experiment_name,
        tags=config.tags,
        save_dir=logger_config.wandb_dir,
    )
    if logger_config.log_hyperparameters:
        logger.experiment.config.update(config.model_dump())

    L_model = TTLightningModel(
        training_config=config.training_config,
        model_config=config.net_config,
        model=model,
        logger_config=config.logger_config,
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
