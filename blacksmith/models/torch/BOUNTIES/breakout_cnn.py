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

        # Offload the categorical math to CPU: on TT the softmax+entropy has poor
        # PCC against CPU, and running it on the host fixes that. We also compute
        # it manually rather than with torch.distributions.Categorical because the
        # manual softmax/gather/entropy gave better precision on TT hardware.
        logits_host = logits.to("cpu").float()
        log_probs_host = torch.log_softmax(logits_host, dim=-1)
        probs_host = log_probs_host.exp()

        if action is None:
            # Sample on CPU with multinomial, then move back to TT.
            action_host = torch.multinomial(probs_host, num_samples=1).squeeze(-1)
            action = action_host.to(target_device)
        else:
            action_host = action.to("cpu")

        log_prob_host = log_probs_host.gather(-1, action_host.unsqueeze(-1)).squeeze(-1)
        entropy_host = -(probs_host * log_probs_host).sum(dim=-1)

        log_prob = log_prob_host.to(target_device)
        entropy = entropy_host.to(target_device)
        return action, log_prob, entropy, self.critic(hidden)
