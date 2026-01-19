# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F

from blacksmith.experiments.torch.mnist.configs import TrainingConfig


class MNISTCNN(nn.Module):
    def __init__(self, config: TrainingConfig):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, config.conv1_channels, config.kernel_size, config.stride, bias=config.bias)
        self.conv2 = nn.Conv2d(
            config.conv1_channels, config.conv2_channels, config.kernel_size, config.stride, bias=config.bias
        )
        self.dropout1 = nn.Dropout(config.dropout1_rate)
        self.dropout2 = nn.Dropout(config.dropout2_rate)
        self.fc1_input_size = 12 * 12 * config.conv2_channels
        self.fc1 = nn.Linear(self.fc1_input_size, config.fc1_size, bias=config.bias)
        self.fc2 = nn.Linear(config.fc1_size, config.output_size, bias=config.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
