# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

from torch.utils.data import DataLoader
from pydantic import BaseModel, Field
from typing import List

from thomas.models.torch.hf_model import LoraModelConfig, load_hf_model
from thomas.data_loaders.hf_lora import LoraDataLoadingConfig, load_data
from thomas.tooling.cli import generate_config, print_trainable_params
from thomas.training.logger_config import LoggerConfig, get_default_logger_config
from thomas.models.torch.loss import TorchLoss
from thomas.models.torch.opt import TorchOptimizer
from thomas.training.pytorch_train.trainer import PyTorchTrainer


# Config model
class LoraTrainingConfig(BaseModel):
    output_dir: str
    epochs: int
    loss: TorchLoss
    optimizer: TorchOptimizer
    optimizer_kwargs: dict
    run_on: str
    save_strategy: str
    save_steps: int
    do_validation: bool


class ExperimentConfig(BaseModel):
    experiment_name: str
    model: LoraModelConfig
    training_config: LoraTrainingConfig
    data_loading: LoraDataLoadingConfig
    logger_config: LoggerConfig = Field(default_factory=get_default_logger_config)
    tags: List[str]


def run_experiment():
    # Load config
    config_path = os.path.splitext(__file__)[0] + ".yaml"
    config = generate_config(ExperimentConfig, config_path)

    # Init model
    model = load_hf_model(config.model)
    model.to(config.training_config.run_on)
    print_trainable_params(model)

    # Load dataset
    # TODO: Custom dataset with methods to retrieve data loaders
    train_dataset, validation_dataset, data_collator, tokenizer = load_data(config.data_loading, config.model.model_id)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=config.data_loading.batch_size,
        pin_memory=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=config.data_loading.batch_size,
        pin_memory=True,
    )

    # Init logger
    # logger_config = config.logger_config

    # Start training
    trainer = PyTorchTrainer(model, train_dataloader, validation_dataloader, config.training_config)
    trainer.train()

    # TODO: Save model


if __name__ == "__main__":
    run_experiment()
