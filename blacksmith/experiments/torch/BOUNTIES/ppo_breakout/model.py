# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class BreakoutCNN(nn.Module):
    # Spatial size after the three conv layers on 84x84 input:
    #   Conv2d(8, stride=4): (84 - 8) / 4 + 1 = 20
    #   Conv2d(4, stride=2): (20 - 4) / 2 + 1 = 9
    #   Conv2d(3, stride=1): ( 9 - 3) / 1 + 1 = 7
    # So the final feature map is 64 * 7 * 7 = 3136.

    PIXEL_SCALE = 255.0

    def __init__(self, num_actions: int, frame_stack: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(frame_stack, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.network(x / self.PIXEL_SCALE))

    def get_action_and_value(self, x: torch.Tensor, action=None):
        hidden = self.network(x / self.PIXEL_SCALE)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits, validate_args=False)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden)
