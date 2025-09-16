# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
import logging
from typing import Optional

import jax
import jax.numpy as jnp
from jax import random
import flax
from flax import linen as nn

# Add the current directory to the path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from blacksmith.tools.cli import generate_config
except ImportError:
    # Fallback for standalone execution
    def generate_config(config_class, config_path):
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return config_class(**config_dict)

from configs import ExperimentConfig, get_cpu_config, get_tt_config
from models.gpt_model import create_model
from datasets.text_dataset import load_text_dataset, create_dataloader
from utils.device_utils import create_device_manager, log_device_info
from utils.training_utils import (
    create_optimizer, create_train_state, training_step, 
    estimate_loss, get_lr, save_checkpoint, load_checkpoint
)
from logging.wandb_utils import init_wandb, log_metrics, finish_wandb


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def initialize_model_and_data(config: ExperimentConfig, device_manager, logger):
    """Initialize model, data, and optimizer."""
    
    # Create model
    model = create_model(config)
    logger.info(f"Created model with config: {config.model_config}")
    
    # Initialize model parameters
    key = random.PRNGKey(config.seed)
    dummy_input = jnp.ones((1, config.model_config.block_size), dtype=jnp.int32)
    
    # Initialize on CPU first (following existing patterns)
    with device_manager.with_device("cpu"):
        params = model.init(key, dummy_input, training=False)
    
    logger.info(f"Initialized model parameters")
    
    # Load dataset
    dataset = load_text_dataset(config)
    dataloader = create_dataloader(dataset, config, device_manager.current_device)
    logger.info(f"Loaded dataset: {config.data_config.dataset}")
    
    # Create optimizer
    optimizer = create_optimizer(config)
    logger.info(f"Created optimizer")
    
    # Create training state
    train_state = create_train_state(model, params, optimizer)
    logger.info(f"Created training state")
    
    return model, dataset, dataloader, train_state


def train_epoch(
    train_state,
    dataloader,
    config: ExperimentConfig,
    device_manager,
    logger,
    wandb_config
):
    """Train for one epoch."""
    
    model = train_state.model
    total_loss = 0.0
    num_batches = 0
    
    # Calculate number of batches per epoch
    # For simplicity, we'll use a fixed number of iterations
    max_iters = min(config.training_config.max_iters, 1000)  # Limit for demo
    
    for step in range(max_iters):
        # Get batch
        try:
            with device_manager.with_device(device_manager.primary_device):
                inputs, targets = dataloader['train']()
        except Exception as e:
            logger.warning(f"Batch loading failed on {device_manager.primary_device}: {e}")
            with device_manager.with_device("cpu"):
                inputs, targets = dataloader['train']()
        
        # Training step
        train_state, loss, logits = training_step(
            train_state, inputs, targets, device_manager
        )
        
        total_loss += loss
        num_batches += 1
        
        # Logging
        if step % config.logging_config.log_every_n_steps == 0:
            current_lr = get_lr(train_state.step, config)
            avg_loss = total_loss / num_batches
            
            logger.info(f"Step {step}: Loss = {loss:.4f}, Avg Loss = {avg_loss:.4f}, LR = {current_lr:.6f}")
            
            if wandb_config and wandb_config.log_on_wandb:
                log_metrics({
                    "train/loss": float(loss),
                    "train/avg_loss": float(avg_loss),
                    "train/learning_rate": float(current_lr),
                    "train/step": step
                }, step=step)
        
        # Validation
        if step % config.training_config.eval_interval == 0 and step > 0:
            val_loss = estimate_loss(
                model, train_state.params, dataloader['val'], 
                config.training_config.eval_iters, device_manager
            )
            
            logger.info(f"Validation loss at step {step}: {val_loss:.4f}")
            
            if wandb_config and wandb_config.log_on_wandb:
                log_metrics({
                    "val/loss": float(val_loss),
                    "val/step": step
                }, step=step)
        
        # Checkpointing
        if step % config.logging_config.checkpoint.save_interval == 0 and step > 0:
            checkpoint_path = os.path.join(
                config.logging_config.checkpoint.checkpoint_dir,
                f"checkpoint_step_{step}.pkl"
            )
            save_checkpoint(train_state, checkpoint_path, step)
            logger.info(f"Saved checkpoint at step {step}")
    
    return train_state


def main():
    """Main training function."""
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting NanoGPT training")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train NanoGPT in JAX")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--device", type=str, choices=["cpu", "tt", "auto"], 
                       default="auto", help="Device to use for training")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = generate_config(ExperimentConfig, args.config)
    else:
        # Use default config based on device
        if args.device == "cpu":
            config = get_cpu_config()
        elif args.device == "tt":
            config = get_tt_config()
        else:
            config = ExperimentConfig()
    
    # Override device setting if specified
    if args.device != "auto":
        config.device_config.primary_device = args.device
    
    logger.info(f"Using configuration: {config}")
    
    # Create device manager
    device_manager = create_device_manager(config)
    log_device_info(device_manager)
    
    # Initialize WandB if enabled
    wandb_config = None
    if config.logging_config.log_on_wandb:
        wandb_config = init_wandb(
            config.logging_config,
            job_type="training",
            dir_path=config.logging_config.checkpoint.checkpoint_dir
        )
        logger.info("Initialized WandB logging")
    
    try:
        # Initialize model and data
        model, dataset, dataloader, train_state = initialize_model_and_data(
            config, device_manager, logger
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            train_state = load_checkpoint(args.resume, train_state)
            logger.info(f"Resumed from checkpoint: {args.resume}")
        
        # Create checkpoint directory
        os.makedirs(config.logging_config.checkpoint.checkpoint_dir, exist_ok=True)
        
        # Training loop
        logger.info("Starting training loop")
        start_time = time.time()
        
        train_state = train_epoch(
            train_state, dataloader, config, device_manager, logger, wandb_config
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Final evaluation
        final_val_loss = estimate_loss(
            model, train_state.params, dataloader['val'],
            config.training_config.eval_iters, device_manager
        )
        
        logger.info(f"Final validation loss: {final_val_loss:.4f}")
        
        if wandb_config and wandb_config.log_on_wandb:
            log_metrics({
                "final/val_loss": float(final_val_loss),
                "final/training_time": float(training_time),
                "final/total_steps": train_state.step
            })
        
        # Save final checkpoint
        final_checkpoint_path = os.path.join(
            config.logging_config.checkpoint.checkpoint_dir,
            "final_checkpoint.pkl"
        )
        save_checkpoint(train_state, final_checkpoint_path, train_state.step)
        logger.info(f"Saved final checkpoint: {final_checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Finish WandB run
        if wandb_config and wandb_config.log_on_wandb:
            finish_wandb()
            logger.info("Finished WandB run")


if __name__ == "__main__":
    main()
