# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.utilities import _scan_checkpoints
from pydantic import BaseModel

from thomas.tooling.forge_tooling import disable_forge_logger
from thomas.training.tt_forge_fe.torch_lightning import TTLightningModel, LightningConfig, GradientCheckpoint
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
    checkpoint_name: str


class CustomLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        self.logged_model_time = set()
        WandbLogger.__init__(self, *args, **kwargs)

    def after_save_checkpoint(self, checkpoint_callback):
        import wandb

        models_to_log = _scan_checkpoints(checkpoint_callback, self.logged_model_time)
        artifact = wandb.Artifact(f"model-{self.experiment.name}", type="model")
        for save_time, model_path, _, tag in models_to_log:
            artifact.add_reference(f"file://{model_path}")
        self.experiment.log_artifact(artifact)
        self.logged_model_time.update({model_path: model_time for model_time, model_path, _, _ in models_to_log})


def test_training(continue_training: bool = False):
    # Currently, forge prints a log on every call of forward and backward, disabling it for now
    disable_forge_logger()

    config: ExperimentConfig = generate_config(ExperimentConfig, "thomas/test/tt-forge-fe/test_mnist.yaml")

    train_loader, test_loader = load_dataset(config.data_loading)
    model = MNISTLinear(config.model)
    logger = CustomLogger(
        project=config.experiment_name,
        tags=config.tags,
        save_dir=config.wandb_dir,
    )
    # Should recover the model from the last checkpoint if needed
    L_model = TTLightningModel(config.lightning, model)

    checkpoint_filename = f"{logger.experiment.name}/{config.checkpoint_name}"

    callbacks = []
    if config.model_checkpoint:
        callbacks.append(
            ModelCheckpoint(
                dirpath=config.checkpoint_dir,
                every_n_train_steps=100,
                filename=checkpoint_filename,
                save_top_k=-1,
            )
        )
    if config.gradiend_checkpoint:
        callbacks.append(GradientCheckpoint())

    trainer = L.Trainer(max_epochs=config.epochs, logger=logger, callbacks=callbacks)
    trainer.fit(L_model, train_loader, test_loader)


if __name__ == "__main__":

    test_training()
