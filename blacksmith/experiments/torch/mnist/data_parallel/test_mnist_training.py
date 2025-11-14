# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import traceback

from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh

from blacksmith.datasets.torch.mnist.dataloader import load_mnist_torch
from blacksmith.tools.cli import generate_config
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.models.torch.mnist.mnist_linear import MNISTLinear
from blacksmith.experiments.torch.mnist.configs import TrainingConfig


def setup_tt_environment(config: TrainingConfig):
    if not config.use_tt:
        return

    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["MESH_SHAPE"] = "1,2"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"

    xr.set_device_type("TT")
    xr.use_spmd()


def mse_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Workaround for nn.MSELoss - it returns a scalar (reduction='mean'),
    # but data parallel operations require loss shape [1, 1] (keepdim=True).
    # github issue: https://github.com/tenstorrent/tt-xla/issues/1993
    loss = (outputs - targets).pow(2)
    loss = loss.mean(dim=1, keepdim=True)
    loss = loss.mean(dim=0, keepdim=True)
    return loss


def setup_mesh(config: TrainingConfig) -> Mesh:
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    axis_names = ("data", "model")
    mesh = Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)
    return mesh


def validate(
    model: torch.nn.Module, val_loader: DataLoader, device: torch.device, logger: TrainingLogger, config: TrainingConfig
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
            loss = mse_loss(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(targets, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    logger.info(f"Validation finished. Avg loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


def train(
    config: TrainingConfig,
    device: torch.device,
    mesh: Mesh,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    logger.info("Starting MNIST training")

    # Load model
    model = MNISTLinear(config.input_size, config.hidden_size, config.output_size, bias=config.bias)
    model = model.to(device)
    logger.info(f"Loaded {config.model_name} model.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    # Load checkpoint if requested
    if config.resume_from_checkpoint:
        checkpoint_manager.load_checkpoint(model, optimizer)

    # Datasets
    train_loader, val_loader = load_mnist_torch(dtype=torch.float32, batch_size=config.batch_size)
    logger.info(f"Train dataset size: {len(train_loader) * config.batch_size}, Eval batches: {len(val_loader)}")

    global_step = 0
    running_loss = 0.0
    # Training
    try:
        model.train()
        for epoch in range(config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{config.num_epochs}")
            for inputs, targets in train_loader:
                inputs = inputs.view(inputs.size(0), -1)
                targets = targets.view(targets.size(0), -1)

                inputs = inputs.to(device)
                targets = targets.to(device)

                # Mark sharding for data parallelism
                xs.mark_sharding(inputs, mesh, ("data", None))
                xs.mark_sharding(targets, mesh, ("data", None))

                # Zero out gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = mse_loss(outputs, targets)

                # Backward pass
                loss.backward()
                running_loss += loss.item()

                # For multichip is better to use xm.optimizer_step - forces execution and ensures correct all-reduce operations
                xm.optimizer_step(optimizer, barrier=True)

                global_step += 1

                # Logging by steps
                if global_step % config.steps_freq == 0:
                    avg_loss = running_loss / config.steps_freq
                    running_loss = 0.0

                    # Run validation and log metrics
                    val_loss, val_acc = validate(model, val_loader, device, logger, config)
                    logger.log_metrics(
                        {"train/loss": avg_loss, "val/loss": val_loss, "val/accuracy": val_acc}, step=global_step
                    )
                    model.train()

                    # Save checkpoint at step
                    if checkpoint_manager.should_save_checkpoint(global_step):
                        checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

            # end epoch loop
            # Save checkpoint at epoch boundary if configured
            if checkpoint_manager.should_save_checkpoint(global_step, epoch):
                checkpoint_manager.save_checkpoint(model, global_step, epoch, optimizer)

        # final model save
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
    # Generate config
    config_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_mnist_training.yaml")
    config: TrainingConfig = generate_config(TrainingConfig, config_file_path)

    # Reproducibility
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Setup TT environment
    setup_tt_environment(config)

    # Compile options
    options = {
        "export_path": "model",
        "export_tensors": True,
        "enable_const_eval": False,
    }
    torch_xla.set_custom_compile_options(options)

    # Setup mesh if using TT
    mesh = None
    if config.use_tt:
        mesh = setup_mesh(config)

    # Logger and checkpoint manager
    logger = TrainingLogger(config)
    checkpoint_manager = CheckpointManager(config, logger)

    # Device
    device = torch_xla.device()

    # Start training
    train(config, device, mesh, logger, checkpoint_manager)
