# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List
from pydantic import BaseModel, Field

from blacksmith.tools.logging.configs import LoggerConfig, get_default_logger_config
from torch_xla.experimental import plugins
import os
import sys


class MNISTLinearConfig(BaseModel):
    input_size: int = 784
    hidden_size: int = 512
    output_size: int = 10
    bias: bool = True


class TrainingConfig(BaseModel):
    train_ratio: float = 0.8
    batch_size: int = 64
    epochs: int = 5
    lr: float = 0.001


class ExperimentConfig(BaseModel):
    device: str = "TT"
    experiment_name: str = "blacksmith-mnist"
    tags: List[str] = ["tt-xla", "model:torch", "plugin", "wandb"]
    net_config: MNISTLinearConfig = Field(default_factory=MNISTLinearConfig)
    loss: str = "torch.nn.MSELoss"
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    data_loading_dtype: str = "bfloat16"
    logger_config: LoggerConfig = Field(default_factory=get_default_logger_config)
