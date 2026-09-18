# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

from pydantic import BaseModel, Field

from blacksmith.tools.logging.configs import LoggerConfig, get_default_logger_config


class MNISTLinearConfig(BaseModel):
    input_size: int = 784
    hidden_size: int = 128
    output_size: int = 10
    bias: bool = True


class TrainingConfig(BaseModel):
    batch_size: int
    epochs: int
    lr: float


class ExperimentConfig(BaseModel):
    experiment_name: str
    tags: List[str]
    net_config: MNISTLinearConfig
    loss: str
    training_config: TrainingConfig
    logger_config: LoggerConfig = Field(default_factory=get_default_logger_config)
    dataset_id: str = "mnist"
    dtype: str = "torch.float32"
