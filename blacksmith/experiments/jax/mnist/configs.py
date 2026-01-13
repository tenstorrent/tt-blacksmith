# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field

from blacksmith.experiments.jax.mnist.logging.logger_config import (
    LoggerConfig,
    get_default_logger_config,
)


class NetConfig(BaseModel):
    input_size: int
    output_size: int


class MNISTLinearConfig(NetConfig):
    input_size: int = 784
    output_size: int = 10
    hidden_size: int = 128


class TrainingConfig(BaseModel):
    batch_size: int = 128
    epochs: int = 10
    lr: float = 0.001
    run_test: bool = True
    export_shlo: bool = False


class EarlyStoppingConfig(BaseModel):
    patience: int = 1
    min_delta: float = 0.001


class ExperimentConfig(BaseModel):
    net_config: MNISTLinearConfig
    training_config: TrainingConfig
    logger_config: LoggerConfig = Field(default_factory=get_default_logger_config)
    early_stopping: EarlyStoppingConfig
