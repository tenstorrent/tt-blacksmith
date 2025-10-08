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
    # Make identical to TT config for fair comparison
    config.model.n_layer = 12
    config.model.n_head = 12
    config.model.n_embd = 768
    config.model.block_size = 1024
    config.data.dataset = "openwebtext"
    config.data.batch_size = 12  # Same as TT config
    config.data.block_size = 1024
    config.training.learning_rate = 6e-4  # Same as TT config
    config.training.max_iters = 600000  # Same as TT config
    config.training.warmup_iters = 2000
    config.training.lr_decay_iters = 600000
    config.training.min_lr = 6e-5
    config.training.eval_interval = 2000
    config.training.eval_iters = 200
    config.training.weight_decay = 1e-1
    config.training.beta1 = 0.9
    config.training.beta2 = 0.95
    config.training.grad_clip = 1.0
    config.training.decay_lr = True
    config.training.log_interval = 1
    config.training.always_save_checkpoint = True
    config.training.device = "cpu"
    config.training.compile = False  # Only difference: disable compilation for CPU
    config.training.dtype = "float32"
    config.device.primary_device = "cpu"  # Only difference: device type
    config.device.enable_fallback = True
    config.device.fallback_device = "cpu"
    config.device.cpu_batch_size = 8
    config.device.tt_batch_size = 12
    config.logging.log_on_wandb = True
    config.logging.wandb_project = "nanogpt-jax-cpu"
    config.logging.wandb_run_name = "nanogpt-cpu-baseline"
    config.logging.checkpoint_dir = "checkpoints_cpu"
    config.logging.save_interval = 2000
    config.logging.max_checkpoints = 3
    config.logging.log_every_n_steps = 10
    config.logging.log_metrics = True
    config.logging.log_gradients = False
    config.early_stopping.enabled = False
    config.early_stopping.patience = 10
    config.early_stopping.min_delta = 0.001
    config.early_stopping.monitor = "val_loss"
    config.experiment_name = "nanogpt-jax-cpu"
    config.seed = 42
    config.resume = False
    config.resume_from_checkpoint = None
    return config


def get_tt_config() -> ExperimentConfig:
    """Get configuration optimized for TT-N150 training."""
    config = get_default_config()
    # Make identical to CPU config for fair comparison
    config.model.n_layer = 12
    config.model.n_head = 12
    config.model.n_embd = 768
    config.model.block_size = 1024
    config.data.dataset = "openwebtext"
    config.data.batch_size = 12  # Same as CPU config
    config.data.block_size = 1024
    config.training.learning_rate = 6e-4  # Same as CPU config
    config.training.max_iters = 600000  # Same as CPU config
    config.training.warmup_iters = 2000
    config.training.lr_decay_iters = 600000
    config.training.min_lr = 6e-5
    config.training.eval_interval = 2000
    config.training.eval_iters = 200
    config.training.weight_decay = 1e-1
    config.training.beta1 = 0.9
    config.training.beta2 = 0.95
    config.training.grad_clip = 1.0
    config.training.decay_lr = True
    config.training.log_interval = 1
    config.training.always_save_checkpoint = True
    config.training.device = "tt"
    config.training.compile = True  # Only difference: enable compilation for TT
    config.training.dtype = "float32"
    config.device.primary_device = "tt"  # Only difference: device type
    config.device.enable_fallback = True
    config.device.fallback_device = "cpu"
    config.device.cpu_batch_size = 8
    config.device.tt_batch_size = 12
    config.logging.log_on_wandb = True
    config.logging.wandb_project = "nanogpt-jax-tt"
    config.logging.wandb_run_name = "nanogpt-tt-n150"
    config.logging.checkpoint_dir = "checkpoints_tt"
    config.logging.save_interval = 2000
    config.logging.max_checkpoints = 3
    config.logging.log_every_n_steps = 10
    config.logging.log_metrics = True
    config.logging.log_gradients = False
    config.early_stopping.enabled = False
    config.early_stopping.patience = 10
    config.early_stopping.min_delta = 0.001
    config.early_stopping.monitor = "val_loss"
    config.experiment_name = "nanogpt-jax-tt"
    config.seed = 42
    config.resume = False
    config.resume_from_checkpoint = None
    return config
