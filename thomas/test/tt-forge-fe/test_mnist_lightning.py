# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from pydantic import BaseModel, Field

from thomas.tooling.forge_tooling import disable_forge_logger
from thomas.training.tt_forge_fe.torch_lightning import (
    TTLightningModel,
    TTLightningConfig,
    GradientCheckpoint,
    TTWandbLogger,
    SaveCheckpointArtifact,
)
from thomas.training.logger_config import LoggerConfig, get_default_logger_config
from thomas.models.torch.mnist_linear import MNISTLinear, ModelConfig
from thomas.tooling.cli import generate_config
from thomas.tooling.data import DataLoadingConfig, load_dataset
from thomas.tooling.forge_tooling import disable_forge_logger
from thomas.training.tt_forge_fe.torch_lightning import TTLightningConfig, TTLightningModel


class ExperimentConfig(BaseModel):
    experiment_name: str
    tags: List[str]
    epochs: int
    model: ModelConfig
    lightning: TTLightningConfig
    data_loading: DataLoadingConfig
    logger_config: LoggerConfig = Field(default_factory=get_default_logger_config)


def test_training():
    # Currently, forge prints a log on every call of forward and backward, disabling it for now
    disable_forge_logger()

    config: ExperimentConfig = generate_config(ExperimentConfig, "thomas/test/tt-forge-fe/test_mnist_lightning.yaml")
    logger_config = config.logger_config

    train_loader, test_loader = load_dataset(config.data_loading)
    model = MNISTLinear(config.model)
    logger = TTWandbLogger(
        project=config.experiment_name,
        tags=config.tags,
        save_dir=logger_config.wandb_dir,
    )
    if logger_config.log_hyperparameters:
        logger.experiment.config.update(config.model_dump())

    L_model = TTLightningModel(config.lightning, model, config.logger_config)

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

    trainer = L.Trainer(max_epochs=config.epochs, logger=logger, callbacks=callbacks)
    trainer.fit(L_model, train_loader, test_loader)


if __name__ == "__main__":
    test_training()
