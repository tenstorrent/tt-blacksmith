# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field
from typing import Optional, List
import os


class ModelConfig(BaseModel):
    """Configuration for the GPT model architecture."""
    # Model dimensions
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab size
    dropout: float = 0.0
    
    # Model type
    bias: bool = False
    use_flash_attn: bool = False


class DataConfig(BaseModel):
    """Configuration for data loading and processing."""
    dataset: str = "openwebtext"
    data_dir: str = "data"
    batch_size: int = 12
    block_size: int = 1024
    
    # Data processing
    num_workers: int = 4
    pin_memory: bool = True


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""
    # Learning rate and optimization
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Learning rate schedule
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    
    # Training control
    eval_interval: int = 2000
    eval_iters: int = 200
    log_interval: int = 1
    always_save_checkpoint: bool = True
    
    # Device configuration
    device: str = "auto"  # "cpu", "tt", or "auto"
    compile: bool = True
    
    # Mixed precision
    dtype: str = "float32"  # "float32", "bfloat16", "float16"


class DeviceConfig(BaseModel):
    """Configuration for device management and fallback."""
    # Primary device
    primary_device: str = "tt"  # "cpu" or "tt"
    
    # Fallback configuration
    enable_fallback: bool = True
    fallback_device: str = "cpu"
    
    # Device-specific settings
    cpu_batch_size: int = 8  # Smaller batch size for CPU fallback
    tt_batch_size: int = 12  # Larger batch size for TT device


class LoggingConfig(BaseModel):
    """Configuration for logging and checkpointing."""
    # WandB configuration
    log_on_wandb: bool = True
    wandb_project: str = "nanogpt-jax-tt"
    wandb_run_name: Optional[str] = None
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 2000
    max_checkpoints: int = 3
    
    # Logging intervals
    log_every_n_steps: int = 10
    log_metrics: bool = True
    log_gradients: bool = False


class EarlyStoppingConfig(BaseModel):
    """Configuration for early stopping."""
    enabled: bool = False
    patience: int = 10
    min_delta: float = 0.001
    monitor: str = "val_loss"


class ExperimentConfig(BaseModel):
    """Main experiment configuration combining all configs."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    device: DeviceConfig
    logging: LoggingConfig
    early_stopping: EarlyStoppingConfig
    
    # Experiment metadata
    experiment_name: str = "nanogpt-jax"
    seed: int = 42
    resume: bool = False
    resume_from_checkpoint: Optional[str] = None


def get_default_config() -> ExperimentConfig:
    """Get default configuration for NanoGPT training."""
    return ExperimentConfig(
        model=ModelConfig(),
        data=DataConfig(),
        training=TrainingConfig(),
        device=DeviceConfig(),
        logging=LoggingConfig(),
        early_stopping=EarlyStoppingConfig()
    )


def get_cpu_config() -> ExperimentConfig:
    """Get configuration optimized for CPU training."""
    config = get_default_config()
    config.device.primary_device = "cpu"
    config.data.batch_size = 4  # Smaller batch size for CPU
    config.training.learning_rate = 3e-4  # Lower learning rate for CPU
    config.training.compile = False  # Disable compilation for CPU
    return config


def get_tt_config() -> ExperimentConfig:
    """Get configuration optimized for TT-N150 training."""
    config = get_default_config()
    config.device.primary_device = "tt"
    config.data.batch_size = 12  # Larger batch size for TT
    config.training.learning_rate = 6e-4  # Standard learning rate
    config.training.compile = True  # Enable compilation for TT
    return config
