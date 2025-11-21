# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import traceback
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torch_xla

from blacksmith.tools.cli import generate_config
from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.torch_xla_utils import setup_tt_environment
from blacksmith.models.torch.mnist.mnist_linear import MNISTLinear
from blacksmith.experiments.torch.mnist.configs import TrainingConfig


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    logger: TrainingLogger,
    config: TrainingConfig,
    loss_fn: torch.nn.Module,
) -> Tuple[float, float]:

    logger.info("Starting validation...")

    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.view(inputs.size(0), -1)
            targets = targets.view(targets.size(0), -1)

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(targets, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    logger.info(f"Validation finished. Avg loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


def train(
    config: TrainingConfig,
    device: torch.device,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting MNIST training (single chip)")

    # Load model
    model = MNISTLinear(config.input_size, config.hidden_size, config.output_size, bias=config.bias)
    model = model.to(device)
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    loss_fn = eval(config.loss_fn)()

    # Datasets
    train_dataset = get_dataset(config, split="train")
    train_loader = train_dataset.get_dataloader()
    val_dataset = get_dataset(config, split="validation")
    val_loader = val_dataset.get_dataloader()
    logger.info(f"Train dataset size: {len(train_loader) * config.batch_size}, Eval batches: {len(val_loader)}")

    # Load checkpoint if requested
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    global_step = 0
    running_loss = 0.0

    try:
        model.train()
        for epoch in range(config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
            for inputs, targets in train_loader:
                inputs = inputs.view(inputs.size(0), -1)
                targets = targets.view(targets.size(0), -1)

                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                # Forward
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                # Backward
                loss.backward()
                running_loss += loss.item()

                optimizer.step()
                torch_xla.sync(wait=True)

                global_step += 1

                # Logging
                if global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq
                    running_loss = 0.0

                    val_loss, val_acc = validate(model, val_loader, device, logger, config, loss_fn)
                    logger.log_metrics(
                        {"train/loss": avg_loss, "val/loss": val_loss, "val/accuracy": val_acc},
                        step=global_step,
                    )
                    model.train()

                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

            # Save checkpoint by epoch boundary
            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        # Final save
        final_checkpoint_path = checkpoint_manager.save_checkpoint(
            model, global_step, config.num_epochs - 1, optimizer, checkpoint_name="final_model.pth"
        )
        logger.log_artifact(final_checkpoint_path, artifact_type="model", name="final_model.pth")
        logger.info("Training finished successfully.")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Training failed with error: {e}", tb)
        raise
    finally:
        logger.finish()


if __name__ == "__main__":
    config_file_path = os.path.join(os.path.dirname(__file__), "test_mnist_training.yaml")
    config: TrainingConfig = generate_config(TrainingConfig, config_file_path)

    # Setup TT environment and device
    if config.use_tt:
        setup_tt_environment(config)
        device = torch_xla.device()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logging + checkpoints
    logger = TrainingLogger(config)
    checkpoint_manager = CheckpointManager(config, logger)

    # Start training
    train(config, device, logger, checkpoint_manager)
