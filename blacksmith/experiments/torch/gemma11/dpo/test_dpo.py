# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma 1.1 2B DPO (Direct Preference Optimization) training script.

Based on the paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/pdf/2305.18290

This script fine-tunes Gemma 1.1 2B using DPO on the math preference dataset.
PEFT method (LoRA, adapters, or full fine-tuning) is configurable via `peft_method`.

Standard DPO Pipeline:
1. First, train an SFT model on the chosen responses
2. Then, use that SFT checkpoint as the reference model (π_ref) for DPO
3. Set `sft_checkpoint_path` in config to point to your SFT checkpoint

If no SFT checkpoint is provided, the base pretrained model is used as π_ref (less ideal).

Model: https://huggingface.co/google/gemma-1.1-2b-it
Dataset: argilla/distilabel-math-preference-dpo
"""
import os
import traceback
from pathlib import Path

import torch
import torch_xla
from tqdm import tqdm

from blacksmith.experiments.torch.gemma11.dpo.configs import DPOTrainingConfig
from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.cli import generate_config
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.dpo_utils import (
    create_reference_model,
    compute_dpo_loss_from_batch,
)


def train_dpo(
    config: DPOTrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    """
    Main DPO training loop for Gemma 1.1 2B.

    Args:
        config: DPO training configuration
        device_manager: Device manager for handling TT/CPU devices
        logger: Training logger for metrics and logging
        checkpoint_manager: Manager for saving/loading checkpoints
    """
    logger.info("Starting Gemma 1.1 2B DPO training...")
    logger.info(f"DPO beta: {config.dpo_beta}")
    logger.info(f"DPO label smoothing: {config.dpo_label_smoothing}")

    # Load policy model (with PEFT if configured)
    policy_model = get_model(config, device_manager.device)
    logger.info(f"Loaded {config.model_name} as policy model.")
    logger.info(f"Policy model parameters: {sum(p.numel() for p in policy_model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in policy_model.parameters() if p.requires_grad)}")

    # Create reference model
    # Standard DPO: π_ref should be an SFT model trained on chosen responses
    if config.sft_checkpoint_path:
        logger.info(f"Loading SFT checkpoint as reference model from: {config.sft_checkpoint_path}")
        # Load checkpoint to CPU first (XLA checkpoints can't be loaded directly to XLA)
        checkpoint = torch.load(config.sft_checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        # Load into policy model (already on device)
        policy_model.load_state_dict(state_dict)
        logger.info("Loaded SFT checkpoint into policy model.")
        # Create reference model as frozen copy of policy
        reference_model = create_reference_model(policy_model)
        logger.info("Created reference model from SFT checkpoint.")
    else:
        logger.warning(
            "No SFT checkpoint provided (sft_checkpoint_path is empty). "
            "Using base pretrained model as reference. "
            "For best results, first train an SFT model on chosen responses."
        )
        reference_model = create_reference_model(policy_model)

    # Freeze reference model
    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.eval()
    reference_model.to(device_manager.device)
    logger.info("Reference model frozen.")

    # Load checkpoint if needed
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint()

    # Load DPO dataset
    train_dataset = get_dataset(config=config, split="train")
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train samples: {len(train_dataset)}")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    global_step = 0
    running_metrics = {
        "loss": 0.0,
        "chosen_rewards": 0.0,
        "rejected_rewards": 0.0,
        "accuracy": 0.0,
    }

    try:
        for epoch in range(config.num_epochs):
            policy_model.train()
            logger.info(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(progress_bar):
                # Check max steps
                if config.max_steps > 0 and global_step >= config.max_steps:
                    logger.info(f"Reached max_steps ({config.max_steps}). Stopping training.")
                    break

                # Zero gradients
                optimizer.zero_grad()

                # Move batch to device
                batch = {k: v.to(device_manager.device) for k, v in batch.items()}

                # Compute DPO loss
                loss, metrics = compute_dpo_loss_from_batch(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    batch=batch,
                    beta=config.dpo_beta,
                    label_smoothing=config.dpo_label_smoothing,
                )

                # Print rewards after each batch
                logger.info(
                    f"[Batch {batch_idx}] Loss: {metrics['loss']:.4f} | "
                    f"Chosen reward: {metrics['chosen_rewards']:.4f} | "
                    f"Rejected reward: {metrics['rejected_rewards']:.4f} | "
                    f"Margin: {metrics['reward_margin']:.4f} | "
                    f"Accuracy: {metrics['accuracy']:.3f}"
                )

                # Accumulate metrics for logging
                for key in running_metrics:
                    running_metrics[key] += metrics[key]

                # Backward pass
                loss.backward()

                if config.use_tt:
                    torch_xla.sync(wait=True)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)

                # Optimizer step - update weights after each batch
                device_manager.optimizer_step(optimizer)

                global_step += 1

                # Logging
                if global_step % config.steps_freq == 0:
                    avg_metrics = {f"dpo/{k}": v / config.steps_freq for k, v in running_metrics.items()}
                    avg_metrics["train/learning_rate"] = config.learning_rate
                    avg_metrics["train/epoch"] = epoch + 1

                    logger.log_metrics(avg_metrics, step=global_step)

                    # Update progress bar
                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_metrics['dpo/loss']:.4f}",
                            "acc": f"{avg_metrics['dpo/accuracy']:.3f}",
                        }
                    )

                    # Reset running metrics
                    for key in running_metrics:
                        running_metrics[key] = 0.0

                # Save checkpoint
                if global_step % config.save_steps == 0:
                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(policy_model, global_step, epoch, optimizer)

            # End of epoch - check max steps
            if config.max_steps > 0 and global_step >= config.max_steps:
                break

            # Save epoch checkpoint
            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(policy_model, global_step, epoch, optimizer)

        # Save final model
        logger.info("Training complete. Saving final model...")
        final_model_path = checkpoint_manager.save_checkpoint(
            policy_model, global_step, epoch, optimizer, checkpoint_name="final_model.pth"
        )
        logger.log_artifact(final_model_path, artifact_type="model", name="final_model.pth")

        logger.info(f"DPO training completed. Total steps: {global_step}")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup
    config_file_path = Path(__file__).parent / "test_dpo.yaml"
    config = generate_config(DPOTrainingConfig, config_file_path)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup
    logger = TrainingLogger(config)

    # Checkpoint manager setup
    checkpoint_manager = CheckpointManager(config, logger)

    # Device setup
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    # Start DPO training
    train_dpo(config, device_manager, logger, checkpoint_manager)
