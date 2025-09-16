# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import wandb
import os
import sys
from typing import Dict, Any, Optional

# Import the actual LoggingConfig from configs.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs import LoggingConfig


def init_wandb(
    config: LoggingConfig,
    job_type: str = "training",
    dir_path: Optional[str] = None,
) -> wandb.Config:
    """Initialize Weights & Biases logging."""
    if not config.log_on_wandb:
        return None
    
    # Set up wandb directory
    if dir_path is None:
        dir_path = config.checkpoint_dir
    
    os.makedirs(dir_path, exist_ok=True)
    
    # Initialize wandb run
    wandb.init(
        project=config.wandb_project,
        entity=None,  # Not in the config structure
        name=config.wandb_run_name,
        tags=[],  # Not in the config structure
        notes=None,  # Not in the config structure
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
