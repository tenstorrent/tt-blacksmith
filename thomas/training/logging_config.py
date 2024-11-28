# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union

from pydantic import BaseModel, Field

from thomas.tooling.cli import generate_config

DEFAULT_LOGGER_CONFIG = "thomas/training/logging_config.yaml"


class CheckpointLoggerConfig(BaseModel):
    log_checkpoint: bool
    checkpoint_dir: str
    checkpoint_name: str
    log_every_n_steps: int
    save_gradients: bool
    save_top_k: int = Field(default=-1)  # how many checkpoints to keep, -1 means keep all


class LoggerConfig(BaseModel):
    checkpoint: CheckpointLoggerConfig
    log_hyperparameters: bool
    wandb_dir: str
    log_train_loss: Union[None, str] = Field(default=None)
    log_train_accuracy: Union[None, str] = Field(default=None)
    log_gradients: Union[None, str] = Field(default=None)
    log_weights: Union[None, str] = Field(default=None)
    log_every_n_steps: int = Field(default=1)
    log_val_loss: Union[None, str] = Field(default=None)
    log_val_accuracy: Union[None, str] = Field(default=None)


def get_default_logger_config() -> LoggerConfig:
    return generate_config(LoggerConfig, DEFAULT_LOGGER_CONFIG)
