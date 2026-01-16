# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    def __init__(self, conv1_channels, conv2_channels, fc1_size, output_size, 
                 dropout1_rate, dropout2_rate, bias):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_channels, 3, 1, bias=bias)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, 3, 1, bias=bias)
        self.dropout1 = nn.Dropout(dropout1_rate)
        self.dropout2 = nn.Dropout(dropout2_rate)
        self.fc1_input_size = 12 * 12 * conv2_channels
        self.fc1 = nn.Linear(self.fc1_input_size, fc1_size, bias=bias)
        self.fc2 = nn.Linear(fc1_size, output_size, bias=bias)
        
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

