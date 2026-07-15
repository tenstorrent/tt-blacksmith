# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional

from pydantic import Field, computed_field

from blacksmith.tools.test_config import TestConfig
from blacksmith.tools.templates.configs import TrainingConfig as BaseTrainingConfig


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
    log_level: str = Field(default="INFO")
    log_interval: int = Field(default=1, gt=0)
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="ALE-Breakout-PPO")
    wandb_run_name: str = Field(default="tt-breakout-ppo")
    wandb_tags: list[str] = Field(default_factory=lambda: ["ppo", "atari", "breakout"])
    wandb_watch_mode: str = Field(default="all")
    wandb_log_freq: int = Field(default=1000)
    model_to_wandb: bool = Field(default=False)

    # Checkpoint settings
    resume_from_checkpoint: bool = Field(default=False)
    resume_option: str = Field(default="last")  # [last, best, path]
    checkpoint_path: str = Field(default="")
    checkpoint_metric: str = Field(default="charts/avg_return")
    checkpoint_metric_mode: str = Field(default="max")
    keep_last_n: int = Field(default=3, ge=0)
    keep_best_n: int = Field(default=3, ge=0)
    save_strategy: str = Field(default="step")
    save_interval: int = Field(default=50, gt=0)
    project_dir: str = Field(default="blacksmith/experiments/torch/BOUNTIES/ppo_breakout")
    save_optim: bool = Field(default=True)
    storage_backend: str = Field(default="local")
    sync_to_storage: bool = Field(default=False)
    load_from_storage: bool = Field(default=False)
    remote_path: str = Field(default="")

    # Reproducibility settings
    seed: int = Field(default=1)
    deterministic: bool = Field(default=True)

    # Device settings
    mesh_shape: Optional[list[int]] = Field(default=None)
    mesh_axis_names: Optional[list[str]] = Field(default=None)

    # Other settings
    framework: str = Field(default="pytorch")
    use_tt: bool = Field(default=True)
    test_config: Optional[TestConfig] = Field(default=None)

    @computed_field
    @property
    def computed_batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @computed_field
    @property
    def minibatch_size(self) -> int:
        return self.computed_batch_size // self.num_minibatches

    @computed_field
    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.computed_batch_size
