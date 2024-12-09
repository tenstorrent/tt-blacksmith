# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel
from thomas.models.config import ModelConfig
from torch import nn


class MNISTLinear(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MNISTLinear, self).__init__()
        self.config = config
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size, bias=config.bias),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size, bias=config.bias),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.output_size, bias=config.bias),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
