# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

from blacksmith.experiments.torch.BOUNTIES.ppo_breakout.configs import TrainingConfig


class RolloutBuffer:
    # Stores transitions collected during a rollout and computes GAE

    def __init__(self, config: TrainingConfig, obs_shape: tuple, device: torch.device):
        self.config = config
        self.pos = 0

        self.obs = torch.zeros((config.num_steps, config.num_envs, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((config.num_steps, config.num_envs), dtype=torch.int32, device=device)
        self.log_probs = torch.zeros((config.num_steps, config.num_envs), device=device)
        self.rewards = torch.zeros((config.num_steps, config.num_envs), device=device)
        self.dones = torch.zeros((config.num_steps, config.num_envs), device=device)
        self.values = torch.zeros((config.num_steps, config.num_envs), device=device)

    def insert(self, obs, action, log_prob, reward, done, value):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.pos = (self.pos + 1) % self.config.num_steps

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor):
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0.0
        for t in reversed(range(self.config.num_steps)):
            if t == self.config.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_val = self.values[t + 1]
            delta = self.rewards[t] + self.config.gamma * next_val * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
        returns = advantages + self.values
        return advantages, returns

    def flatten(self):
        # Flatten (num_steps, num_envs, ...) -> (batch_size, ...)
        return (
            self.obs.reshape(-1, *self.obs.shape[2:]),
            self.actions.reshape(-1),
            self.log_probs.reshape(-1),
            self.values.reshape(-1),
        )
