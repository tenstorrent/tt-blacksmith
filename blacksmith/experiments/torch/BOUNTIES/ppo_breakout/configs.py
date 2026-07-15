# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import Field, computed_field, model_validator

from blacksmith.tools.templates.configs import TrainingConfig as BaseTrainingConfig
from blacksmith.tools.test_config import TestConfig


class TrainingConfig(BaseTrainingConfig):
    # Environment settings
    env_id: str = Field(default="ALE/Breakout-v5")
    num_envs: int = Field(default=8, gt=0)
    frame_stack: int = Field(default=4, gt=0)
    frame_skip: int = Field(default=4, gt=0)

    # Training hyperparameters
    total_timesteps: int = Field(default=10_000_000, gt=0)
    learning_rate: float = Field(default=2.5e-4, gt=0)
    anneal_lr: bool = Field(default=True)
    batch_size: int = Field(default=1024, gt=0)  # num_envs * num_steps (8 * 128)

    # PPO hyperparameters
    num_steps: int = Field(default=128, gt=0)
    gamma: float = Field(default=0.99)
    gae_lambda: float = Field(default=0.95)
    num_minibatches: int = Field(default=4, gt=0)
    update_epochs: int = Field(default=4, gt=0)
    clip_coef: float = Field(default=0.1)
    norm_adv: bool = Field(default=True)
    clip_vloss: bool = Field(default=True)
    ent_coef: float = Field(default=0.01)
    vf_coef: float = Field(default=0.5)
    max_grad_norm: float = Field(default=0.5)

    # Logging settings
    log_interval: int = Field(default=1, gt=0)
    wandb_project: str = Field(default="ALE-Breakout-PPO")
    wandb_run_name: str = Field(default="tt-breakout-ppo")
    wandb_tags: list[str] = Field(default_factory=lambda: ["ppo", "atari", "breakout"])

    # Checkpoint settings
    checkpoint_metric: str = Field(default="charts/avg_return")
    checkpoint_metric_mode: str = Field(default="max")
    save_strategy: str = Field(default="step")
    save_interval: int = Field(default=50, gt=0)
    project_dir: str = Field(default="blacksmith/experiments/torch/BOUNTIES/ppo_breakout")
    save_optim: bool = Field(default=True)

    # Reproducibility settings
    seed: int = Field(default=1)
    deterministic: bool = Field(default=True)

    # Device settings
    mesh_shape: Optional[list[int]] = Field(default=None)
    mesh_axis_names: Optional[list[str]] = Field(default=None)

    # Other settings
    test_config: Optional[TestConfig] = Field(default=None)

    @model_validator(mode="after")
    def _sync_batch_size(self):
        object.__setattr__(self, "batch_size", self.num_envs * self.num_steps)
        return self

    @computed_field
    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @computed_field
    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.batch_size
