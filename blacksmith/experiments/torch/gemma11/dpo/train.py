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

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.gemma11.dpo.configs import DPOTrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.dpo_utils import (
    compute_dpo_loss_from_batch,
    create_reference_model,
)
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager


def validate_dpo(
    policy_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    val_dataloader,
    config: DPOTrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
) -> dict[str, float]:
    """Evaluate DPO accuracy and average reward margin on the validation set."""
    logger.info("\n=== Starting DPO Validation ===")
    policy_model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_reward_margin = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            batch = {k: v.to(device_manager.device) for k, v in batch.items()}
            _, metrics = compute_dpo_loss_from_batch(
                policy_model=policy_model,
                reference_model=reference_model,
                batch=batch,
                beta=config.dpo_beta,
                label_smoothing=config.dpo_label_smoothing,
                use_tt=config.use_tt,
            )
            total_loss += metrics["loss"]
            total_accuracy += metrics["accuracy"]
            total_reward_margin += metrics["reward_margin"]
            num_batches += 1

            if config.use_tt:
                torch_xla.sync(wait=True)

    policy_model.train()

    if num_batches == 0:
        logger.warning("Validation dataloader is empty; skipping validation metrics.")
        return {"val/loss": 0.0, "val/accuracy": 0.0, "val/reward_margin": 0.0}

    val_metrics = {
        "val/loss": total_loss / num_batches,
        "val/accuracy": total_accuracy / num_batches,
        "val/reward_margin": total_reward_margin / num_batches,
    }
    logger.info(
        f"Validation | loss: {val_metrics['val/loss']:.4f} | "
        f"accuracy: {val_metrics['val/accuracy']:.3f} | "
        f"reward_margin: {val_metrics['val/reward_margin']:.4f}"
    )
    return val_metrics


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
    total_params = sum(p.numel() for p in policy_model.parameters())
    trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    logger.info(f"Loaded {config.model_name} as policy model.")
    logger.info(f"Policy model parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    logger.log_model_info(
        {
            "model_name": config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "dpo_beta": config.dpo_beta,
            "sft_checkpoint_path": config.sft_checkpoint_path or "none",
        }
    )
    logger.watch_model(policy_model)

    # Create reference model
    # Standard DPO: π_ref should be an SFT model trained on chosen responses
    if config.sft_checkpoint_path:
        logger.info(f"Loading SFT checkpoint as reference model from: {config.sft_checkpoint_path}")
        checkpoint_manager.load_checkpoint_path(config.sft_checkpoint_path, policy_model)
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

    val_dataloader = None
    if config.do_validation:
        val_dataset = get_dataset(config=config, split="validation")
        val_dataloader = val_dataset.get_dataloader()
        logger.info(f"Loaded {config.dataset_id} validation split. Val samples: {len(val_dataset)}")

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
        "reward_margin": 0.0,
        "accuracy": 0.0,
    }
    last_step_metrics = {}
    last_val_metrics = {}

    try:
        if config.do_validation and val_dataloader is not None:
            last_val_metrics = validate_dpo(
                policy_model, reference_model, val_dataloader, config, device_manager, logger
            )
            logger.log_metrics({**last_val_metrics, "train/epoch": 0}, commit=True, step=global_step)

        for epoch in range(config.num_epochs):
            policy_model.train()
            logger.info(f"\n=== Epoch {epoch + 1}/{config.num_epochs} ===")

            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            num_batches = len(train_dataloader)
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
                    use_tt=config.use_tt,
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

                step_metrics = {}
                if global_step % config.steps_freq == 0:
                    avg_metrics = {f"dpo/{k}": v / config.steps_freq for k, v in running_metrics.items()}
                    avg_metrics["train/learning_rate"] = config.learning_rate
                    avg_metrics["train/epoch"] = epoch + 1
                    last_step_metrics = avg_metrics
                    step_metrics.update(avg_metrics)

                    progress_bar.set_postfix(
                        {
                            "loss": f"{avg_metrics['dpo/loss']:.4f}",
                            "acc": f"{avg_metrics['dpo/accuracy']:.3f}",
                        }
                    )

                    for key in running_metrics:
                        running_metrics[key] = 0.0

                if config.do_validation and val_dataloader is not None:
                    is_val_step = global_step % config.val_steps_freq == 0
                    is_end_of_epoch = batch_idx == num_batches - 1
                    if is_val_step or (is_end_of_epoch and not is_val_step):
                        last_val_metrics = validate_dpo(
                            policy_model, reference_model, val_dataloader, config, device_manager, logger
                        )
                        step_metrics.update(last_val_metrics)
                        step_metrics["train/epoch"] = epoch + 1

                logger.log_metrics(step_metrics, commit=True, step=global_step)

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

        logger.log_summary(
            {
                "total_steps": global_step,
                "final_epoch": epoch + 1,
                **{k: v for k, v in last_step_metrics.items()},
                **{k: v for k, v in last_val_metrics.items()},
            }
        )
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