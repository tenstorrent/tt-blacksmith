# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torch import nn


class MNISTLinear(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias=True):
        super(MNISTLinear, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=bias),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
