# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field
from thomas.models.jax.mnist.mnist_linear import MNISTLinearConfig
from thomas.experiments.jax.mnist.logging.logger_config import LoggerConfig, get_default_logger_config


class TrainingConfig(BaseModel):
    batch_size: int = 128
    epochs: int = 10
    lr: float = 0.001


class ExperimentConfig(BaseModel):
    net_config: MNISTLinearConfig
    training_config: TrainingConfig
    logger_config: LoggerConfig = Field(default_factory=get_default_logger_config)
