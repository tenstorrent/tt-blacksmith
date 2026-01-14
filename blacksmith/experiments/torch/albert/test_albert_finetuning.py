# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import traceback
from pathlib import Path

import torch
import torch_xla
from torch.utils.data import DataLoader
from tqdm import tqdm

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.experiments.torch.albert.configs import TrainingConfig
from blacksmith.models.torch.huggingface.albert import AlbertWithMLP
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager


def validate(
    model: torch.nn.Module,
    val_data_loader: DataLoader,
    logger: TrainingLogger,
    device_manager: DeviceManager,
    loss_fn: torch.nn.Module,
) -> float:
    logger.info("Starting validation...")
    model.eval()

    total_val_loss = 0.0
    num_val_batches = 0

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validation"):
            batch = device_manager.prepare_batch(batch)

            # Forward pass
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            predictions = torch.argmax(logits, dim=-1)

            # Compute loss
            loss = loss_fn(logits, batch["labels"])
            total_val_loss += loss.item()

            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0.0
    metrics = {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}

    return avg_val_loss, metrics


def train(
    config: TrainingConfig, device_manager: DeviceManager, logger: TrainingLogger, checkpoint_manager: CheckpointManager
):
    logger.info("Starting training...")

    # Load model
    model = AlbertWithMLP(config)
    model.to(eval(config.dtype))
    model.to(device_manager.device)
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Load checkpoint if needed
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint()

    # Load dataset
    train_dataset = get_dataset(config=config, split="train")
    train_dataloader = train_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Train dataset size: {len(train_dataloader)*config.batch_size}")

    eval_dataset = get_dataset(config=config, split="test")
    eval_dataloader = eval_dataset.get_dataloader()
    logger.info(f"Loaded {config.dataset_id} dataset. Eval dataset size: {len(eval_dataloader)*config.batch_size}")

    # Init training components (optimizer, lr scheduler, etc.)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    global_step = 0
    running_loss = 0.0
    model.train()
    try:
        for epoch in range(config.num_epochs):
            for batch in tqdm(train_dataloader):
                optimizer.zero_grad()

                batch = device_manager.prepare_batch(batch)

                # Forward pass
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

                # Compute loss
                loss = loss_fn(logits, batch["labels"])
                running_loss += loss.item()

                # Backward pass
                loss.backward()
                if config.use_tt:
                    torch_xla.sync(wait=True)

                # Update parameters
                device_manager.optimizer_step(optimizer)

                global_step += 1
                if global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq
                    logger.log_metrics({"train/loss": avg_loss}, commit=False, step=global_step)
                    running_loss = 0.0

                    # Do validation
                    valid_loss, metrics = validate(model, eval_dataloader, logger, device_manager.device, loss_fn)
                    logger.log_metrics({"val/loss": valid_loss, "val/accuracy": metrics["accuracy"]}, step=global_step)
                    model.train()

                    # Save checkpoint
                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        # Save final model
        final_model_path = checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)
        logger.log_artifact(final_model_path, artifact_type="model", name="final_model.pth")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    # Config setup
    default_config = Path(__file__).parent / "test_albert_finetuning.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config)

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

    # Start training
    train(config, device_manager, logger, checkpoint_manager)
