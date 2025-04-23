# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Union

from pydantic import BaseModel, Field, model_validator

from blacksmith.tools.cli import generate_config


class CheckpointLoggerConfig(BaseModel):
    log_checkpoint: bool
    checkpoint_dir: str
    checkpoint_name: str
    log_every_n_steps: Union[None, int] = Field(default=None)
    log_every_n_epochs: Union[None, int] = Field(default=None)
    save_gradients: bool
    save_top_k: int = Field(default=-1)  # how many checkpoints to keep, -1 means keep all

    @model_validator(mode="after")
    def check_exclusive_every_n(self):
        if self.log_every_n_steps is not None and self.log_every_n_epochs is not None:
            raise ValueError("log_every_n_steps and log_every_n_epochs are mutually exclusive")
        return self


class LoggerConfig(BaseModel):
    checkpoint: Union[None, CheckpointLoggerConfig] = Field(default=None)
    log_hyperparameters: bool
    experiment_name: str
    wandb_dir: str
    log_train_loss: Union[None, str] = Field(default=None)
    log_train_accuracy: Union[None, str] = Field(default=None)
    log_gradients: Union[None, str] = Field(default=None)
    log_weights: Union[None, str] = Field(default=None)
    log_optimizer: Union[None, str] = Field(default=None)
    log_every_n_steps: Union[None, int] = Field(default=None)
    log_every_n_epochs: Union[None, int] = Field(default=None)
    log_val_loss: Union[None, str] = Field(default=None)
    log_val_accuracy: Union[None, str] = Field(default=None)

    @model_validator(mode="after")
    def check_exclusive_every_n(self):
        if self.log_every_n_steps is not None and self.log_every_n_epochs is not None:
            raise ValueError("log_every_n_steps and log_every_n_epochs are mutually exclusive")
        return self


def get_default_logger_config() -> LoggerConfig:
    logger_config = os.path.join(os.path.dirname(__file__), "logger_config.yaml")
    return generate_config(LoggerConfig, logger_config)
