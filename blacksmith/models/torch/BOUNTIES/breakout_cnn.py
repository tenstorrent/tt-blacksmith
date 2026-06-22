# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn


def layer_init(layer, std=2.0**0.5, bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class BreakoutCNN(nn.Module):
    # Spatial size after the three conv layers on 84x84 input:
    #   Conv2d(8, stride=4): (84 - 8) / 4 + 1 = 20
    #   Conv2d(4, stride=2): (20 - 4) / 2 + 1 = 9
    #   Conv2d(3, stride=1): ( 9 - 3) / 1 + 1 = 7
    # So the final feature map is 64 * 7 * 7 = 3136.

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
        return self.critic(self.network(x))

    def get_action_and_value(self, x: torch.Tensor, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        target_device = logits.device
        num_actions = logits.shape[-1]

        log_probs = torch.log_softmax(logits, dim=-1).to(target_device)
        probs = log_probs.exp()

        if action is None:
            # Sample on device via the Gumbel-max trick instead of torch.multinomial, to
            # keep a random node out of the graph: re-drawn across executions it would
            # desync action from its log_prob
            u = torch.rand(log_probs.shape, device="cpu").to(target_device)
            gumbel = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            action = (log_probs + gumbel).argmax(dim=-1)

        # Select the chosen action's log-prob with a one-hot masked sum instead
        # of log_probs.gather(-1, action): torch.gather returns wrong values.
        onehot = torch.nn.functional.one_hot(action.long(), num_actions).to(log_probs.dtype)
        log_prob = (log_probs * onehot).sum(dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return action, log_prob, entropy, self.critic(hidden)
