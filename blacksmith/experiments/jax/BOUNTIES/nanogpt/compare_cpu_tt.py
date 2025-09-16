# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Comparison script for CPU vs TT-N150 training results.
This script runs training on both devices and compares the results.
"""

import os
import sys
import time
import logging
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from configs import get_cpu_config, get_tt_config
from models.gpt_model import create_model
from datasets.text_dataset import load_text_dataset, create_dataloader
from utils.device_utils import create_device_manager
from utils.training_utils import (
    create_optimizer, create_train_state, training_step, 
    estimate_loss, get_lr
)
from wandb_logging.wandb_utils import init_wandb, log_metrics, finish_wandb


def run_training(config, device_name, max_steps=100):
    """Run training for a specified number of steps."""
    print(f"\n=== Running {device_name} Training ===")
    
    # Setup
    device_manager = create_device_manager(config)
    model = create_model(config)
    
    # Initialize model
    key = jax.random.PRNGKey(config.seed)
    dummy_input = jnp.ones((1, config.model_config.block_size), dtype=jnp.int32)
    
    with device_manager.with_device("cpu"):
        params = model.init(key, dummy_input, training=False)
    
    # Load data
    dataset = load_text_dataset(config)
    dataloader = create_dataloader(dataset, config, device_manager.current_device)
    
    # Create optimizer and training state
    optimizer = create_optimizer(config)
    train_state = create_train_state(model, params, optimizer)
    
    # Training loop
    losses = []
    learning_rates = []
    val_losses = []
    steps = []
    
    start_time = time.time()
    
    for step in range(max_steps):
        # Get batch
        try:
            with device_manager.with_device(device_manager.primary_device):
                inputs, targets = dataloader['train']()
        except Exception as e:
            with device_manager.with_device("cpu"):
                inputs, targets = dataloader['train']()
        
        # Training step
        train_state, loss, logits = training_step(
            train_state, inputs, targets, device_manager
        )
        
        # Record metrics
        current_lr = get_lr(train_state.step, config)
        losses.append(float(loss))
        learning_rates.append(float(current_lr))
        steps.append(step)
        
        # Validation
        if step % 20 == 0 and step > 0:
            val_loss = estimate_loss(
                model, train_state.params, dataloader['val'], 
                5, device_manager  # Use fewer eval iterations for speed
            )
            val_losses.append(float(val_loss))
            print(f"Step {step}: Loss={loss:.4f}, Val Loss={val_loss:.4f}, LR={current_lr:.6f}")
        else:
            print(f"Step {step}: Loss={loss:.4f}, LR={current_lr:.6f}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"✓ {device_name} training completed in {training_time:.2f} seconds")
    
    return {
        'device': device_name,
        'losses': losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'steps': steps,
        'training_time': training_time,
        'final_loss': losses[-1] if losses else 0.0,
        'final_val_loss': val_losses[-1] if val_losses else 0.0
    }


def plot_comparison(cpu_results, tt_results):
    """Plot comparison of CPU vs TT-N150 results."""
    print("\n=== Generating Comparison Plots ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss comparison
    ax1.plot(cpu_results['steps'], cpu_results['losses'], 'b-', label='CPU', linewidth=2)
    ax1.plot(tt_results['steps'], tt_results['losses'], 'r-', label='TT-N150', linewidth=2)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation loss comparison
    cpu_val_steps = [cpu_results['steps'][i] for i in range(0, len(cpu_results['steps']), 20) if i > 0]
    tt_val_steps = [tt_results['steps'][i] for i in range(0, len(tt_results['steps']), 20) if i > 0]
    
    ax2.plot(cpu_val_steps, cpu_results['val_losses'], 'b-o', label='CPU', linewidth=2)
    ax2.plot(tt_val_steps, tt_results['val_losses'], 'r-o', label='TT-N150', linewidth=2)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate comparison
    ax3.plot(cpu_results['steps'], cpu_results['learning_rates'], 'b-', label='CPU', linewidth=2)
    ax3.plot(tt_results['steps'], tt_results['learning_rates'], 'r-', label='TT-N150', linewidth=2)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance comparison
    devices = ['CPU', 'TT-N150']
    training_times = [cpu_results['training_time'], tt_results['training_time']]
    final_losses = [cpu_results['final_loss'], tt_results['final_loss']]
    
    ax4_twin = ax4.twinx()
    bars1 = ax4.bar([x - 0.2 for x in range(len(devices))], training_times, 0.4, 
                    label='Training Time (s)', color='skyblue', alpha=0.7)
    bars2 = ax4_twin.bar([x + 0.2 for x in range(len(devices))], final_losses, 0.4, 
                         label='Final Loss', color='lightcoral', alpha=0.7)
    
    ax4.set_xlabel('Device')
    ax4.set_ylabel('Training Time (seconds)', color='blue')
    ax4_twin.set_ylabel('Final Loss', color='red')
    ax4.set_title('Performance Comparison')
    ax4.set_xticks(range(len(devices)))
    ax4.set_xticklabels(devices)
    
    # Add value labels on bars
    for i, (time_val, loss_val) in enumerate(zip(training_times, final_losses)):
        ax4.text(i - 0.2, time_val + 0.1, f'{time_val:.1f}s', ha='center', va='bottom')
        ax4_twin.text(i + 0.2, loss_val + 0.01, f'{loss_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('cpu_tt_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Comparison plot saved as 'cpu_tt_comparison.png'")
    
    return fig


def print_comparison_summary(cpu_results, tt_results):
    """Print detailed comparison summary."""
    print("\n" + "=" * 60)
    print("CPU vs TT-N150 Training Comparison Summary")
    print("=" * 60)
    
    print(f"{'Metric':<25} {'CPU':<15} {'TT-N150':<15} {'Difference':<15}")
    print("-" * 60)
    
    # Training time
    time_diff = tt_results['training_time'] - cpu_results['training_time']
    time_ratio = tt_results['training_time'] / cpu_results['training_time']
    print(f"{'Training Time (s)':<25} {cpu_results['training_time']:<15.2f} {tt_results['training_time']:<15.2f} {time_diff:<15.2f}")
    
    # Final training loss
    loss_diff = tt_results['final_loss'] - cpu_results['final_loss']
    print(f"{'Final Training Loss':<25} {cpu_results['final_loss']:<15.4f} {tt_results['final_loss']:<15.4f} {loss_diff:<15.4f}")
    
    # Final validation loss
    val_loss_diff = tt_results['final_val_loss'] - cpu_results['final_val_loss']
    print(f"{'Final Val Loss':<25} {cpu_results['final_val_loss']:<15.4f} {tt_results['final_val_loss']:<15.4f} {val_loss_diff:<15.4f}")
    
    # Loss convergence
    cpu_loss_range = max(cpu_results['losses']) - min(cpu_results['losses'])
    tt_loss_range = max(tt_results['losses']) - min(tt_results['losses'])
    print(f"{'Loss Range':<25} {cpu_loss_range:<15.4f} {tt_loss_range:<15.4f} {tt_loss_range - cpu_loss_range:<15.4f}")
    
    print("\n" + "=" * 60)
    print("Analysis:")
    
    # Performance analysis
    if time_ratio < 1.0:
        print(f"✓ TT-N150 is {1/time_ratio:.2f}x faster than CPU")
    else:
        print(f"⚠ TT-N150 is {time_ratio:.2f}x slower than CPU")
    
    # Loss analysis
    loss_diff_pct = abs(loss_diff) / cpu_results['final_loss'] * 100
    if loss_diff_pct < 5.0:
        print(f"✓ Loss values are within 5% difference ({loss_diff_pct:.1f}%)")
    else:
        print(f"⚠ Loss values differ by {loss_diff_pct:.1f}%")
    
    # Convergence analysis
    if abs(val_loss_diff) < 0.1:
        print("✓ Validation losses are very close, indicating good parity")
    else:
        print("⚠ Validation losses show significant difference")


def main():
    """Main comparison function."""
    print("NanoGPT CPU vs TT-N150 Training Comparison")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    max_steps = 100  # Reduced for faster comparison
    
    try:
        # Run CPU training
        cpu_config = get_cpu_config()
        cpu_results = run_training(cpu_config, "CPU", max_steps)
        
        # Run TT-N150 training
        tt_config = get_tt_config()
        tt_results = run_training(tt_config, "TT-N150", max_steps)
        
        # Generate comparison
        plot_comparison(cpu_results, tt_results)
        print_comparison_summary(cpu_results, tt_results)
        
        print("\n✓ Comparison completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
