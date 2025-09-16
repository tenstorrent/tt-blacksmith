# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import wandb
import os
from typing import Dict, Any, Optional
from wandb_logging.logger_config import LoggerConfig


def init_wandb(
    config: LoggerConfig,
    job_type: str = "training",
    dir_path: Optional[str] = None,
) -> wandb.Config:
    """Initialize Weights & Biases logging."""
    if not config.log_on_wandb:
        return None
    
    # Set up wandb directory
    if dir_path is None:
        dir_path = config.checkpoint.checkpoint_dir
    
    os.makedirs(dir_path, exist_ok=True)
    
    # Initialize wandb run
    wandb.init(
        project=config.wandb_config.project,
        entity=config.wandb_config.entity,
        name=config.wandb_config.run_name,
        tags=config.wandb_config.tags,
        notes=config.wandb_config.notes,
        job_type=job_type,
        dir=dir_path,
        resume="allow",
    )
    
    return wandb.config


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to WandB."""
    if wandb.run is not None:
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)


def log_model_weights(model_params: Any, step: int) -> None:
    """Log model weights to WandB."""
    if wandb.run is not None:
        # Convert JAX parameters to a format suitable for logging
        # This is a simplified version - in practice, you might want to
        # log specific layers or use wandb.watch() for automatic logging
        pass


def finish_wandb() -> None:
    """Finish WandB run."""
    if wandb.run is not None:
        wandb.finish()


def watch_model(model, log_freq: int = 100) -> None:
    """Watch model for automatic gradient and parameter logging."""
    if wandb.run is not None:
        wandb.watch(model, log="all", log_freq=log_freq)
