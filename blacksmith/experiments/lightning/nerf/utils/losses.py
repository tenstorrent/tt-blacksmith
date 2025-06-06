# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction="mean")

    def forward(self, inputs, targets):
        loss = 0.0
        if "rgb_coarse" in inputs:
            loss += self.loss(inputs["rgb_coarse"], targets)
        if "rgb_fine" in inputs:
            loss += self.loss(inputs["rgb_fine"], targets)

        return loss


loss_dict = {"mse": MSELoss}
