# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pydantic import BaseModel

from thomas.tooling.forge_tooling import disable_forge_logger
from thomas.training.tt_forge_fe.torch_lightning import (
    TTLightningModel,
    LightningConfig,
    GradCheckpoint
)
from thomas.models.torch.mnist_linear import MNISTLinear, ModelConfig
from thomas.tooling.cli import generate_config
from thomas.tooling.data import DataLoadingConfig, load_dataset
from thomas.tooling.forge_tooling import disable_forge_logger
from thomas.training.tt_forge_fe.torch_lightning import LightningConfig, TTLightningModel


class ExperimentConfig(BaseModel):
    experiment_name: str
    tags: List[str]
    epochs: int
    wandb_dir: str
    checkpoint_dir: str
    model: ModelConfig
    lightning: LightningConfig
    data_loading: DataLoadingConfig
    gradiend_checkpoint: bool
    model_checkpoint: bool

class CustomLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        WandbLogger.__init__(self, *args, **kwargs)
    
    def after_save_checkpoint(self, checkpoint_callback):
        # print all atributes of self
        checkpoint_path = checkpoint_callback.format_checkpoint_name({"epoch": self.current_epoch, "step": self.global_step})
        import wandb
        artifact = wandb.Artifact("model", type="model")
        print(f"Adding {checkpoint_path} to artifact")
        import os
        assert os.path.exists(checkpoint_path)
        artifact.add_reference(f"file://{checkpoint_path}")


def test_training():
    # Currently, forge prints a log on every call of forward and backward, disabling it for now
    disable_forge_logger()

    config: ExperimentConfig = generate_config(ExperimentConfig, "thomas/test/tt-forge-fe/test_mnist.yaml")

    train_loader, test_loader = load_dataset(config.data_loading)
    model = MNISTLinear(config.model)
    tag = "tt-forge"
    log_model = False
    logger = CustomLogger(
        project=config.experiment_name,
        log_model=log_model,
        tags=[tag],
        save_dir=config.wandb_dir,
    )
    L_model = TTLightningModel(config.lightning, model)

    checkpoint_filename = logger.experiment.name + "/{epoch:02d}-{step:06d}"
    
    callbacks = []
    if config.model_checkpoint:
        callbacks.append(ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            every_n_train_steps=100,
            filename=checkpoint_filename,
            save_top_k=-1,
        ))
    if config.gradiend_checkpoint:
        callbacks.append(GradCheckpoint())

    trainer = L.Trainer(max_epochs=config.epochs, logger=logger, callbacks=callbacks)
    trainer.fit(L_model, train_loader, test_loader)


if __name__ == "__main__":
    test_training()
