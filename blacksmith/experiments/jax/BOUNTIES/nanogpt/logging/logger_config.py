# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from typing import Optional
import os


class CheckpointConfig(BaseModel):
    """Configuration for model checkpointing."""
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 2000
    max_checkpoints: int = 3
    save_best_only: bool = True
    monitor: str = "val_loss"
    mode: str = "min"


class WandBConfig(BaseModel):
    """Configuration for Weights & Biases logging."""
    project: str = "nanogpt-jax-tt"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: list = Field(default_factory=list)
    notes: Optional[str] = None


class LoggerConfig(BaseModel):
    """Main logging configuration."""
    # WandB settings
    log_on_wandb: bool = True
    wandb_config: WandBConfig = Field(default_factory=WandBConfig)
    
    # Checkpointing
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    
    # Logging intervals
    log_every_n_steps: int = 10
    log_metrics: bool = True
    log_gradients: bool = False
    log_model_weights: bool = False
    
    # Console logging
    log_to_console: bool = True
    log_level: str = "INFO"


def get_default_logger_config() -> LoggerConfig:
    """Get default logger configuration."""
    return LoggerConfig()


def get_cpu_logger_config() -> LoggerConfig:
    """Get logger configuration for CPU training."""
    config = get_default_logger_config()
    config.wandb_config.project = "nanogpt-jax-cpu"
    config.wandb_config.run_name = "nanogpt-cpu-baseline"
    return config


def get_tt_logger_config() -> LoggerConfig:
    """Get logger configuration for TT-N150 training."""
    config = get_default_logger_config()
    config.wandb_config.project = "nanogpt-jax-tt"
    config.wandb_config.run_name = "nanogpt-tt-n150"
    return config
