# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

from pydantic import BaseModel, Field
from typing import List
from transformers import TrainingArguments, Trainer

from thomas.models.torch.hf_model import LoraModelConfig, load_hf_model
from thomas.data_loaders.instruction_tuning import InstructionTuningDataLoadingConfig, InstructionTuningDataStore
from thomas.tooling.cli import generate_config, print_trainable_params
from thomas.models.torch.loss import TorchLoss
from thomas.training.logger_config import LoggerConfig, get_default_logger_config


# Config model
class LoraTrainingConfig(BaseModel):
    output_dir: str
    epochs: int
    loss: TorchLoss
    lr: float
    run_on: str
    eval_strategy: str
    save_strategy: str
    save_steps: int


class ExperimentConfig(BaseModel):
    experiment_name: str
    model: LoraModelConfig
    training_config: LoraTrainingConfig
    data_loading: InstructionTuningDataLoadingConfig
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
    data_store = InstructionTuningDataStore(config.data_loading, config.model.model_id)

    # Training loop
    train_args = TrainingArguments(
        output_dir=config.training_config.output_dir,
        num_train_epochs=config.training_config.epochs,
        per_device_train_batch_size=config.data_loading.batch_size,
        learning_rate=config.training_config.lr,
        remove_unused_columns=False,
        eval_strategy=config.training_config.eval_strategy,
        save_strategy=config.training_config.save_strategy,
        save_steps=config.logger_config.log_every_n_steps,
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=data_store.train_set,
        eval_dataset=data_store.validation_set,
        tokenizer=data_store.tokenizer,
        data_collator=data_store.data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    run_experiment()
