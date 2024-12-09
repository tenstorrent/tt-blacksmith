# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field
from typing import List
from thomas.models.config import ModelConfig
from thomas.models.torch.dtypes import TorchDType
from thomas.tooling.config import DataLoadingConfig
from thomas.training.config import TrainingConfiguration
from thomas.training.logger_config import LoggerConfig, get_default_logger_config


class ExperimentConfig(BaseModel):
    experiment_name: str
    tags: List[str]
    model: ModelConfig
    training: TrainingConfiguration
    data_loading: DataLoadingConfig
    logger_config: LoggerConfig = Field(default_factory=get_default_logger_config)
