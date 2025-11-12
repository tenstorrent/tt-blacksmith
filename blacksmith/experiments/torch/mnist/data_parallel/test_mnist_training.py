# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import NoneType
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from blacksmith.tools.cli import generate_config
from blacksmith.models.torch.mnist.mnist_linear import MNISTLinear
from blacksmith.experiments.torch.mnist.configs import ExperimentConfig
import os
import wandb


def setup_tt_environment():
    """Setup TensorTrent environment and plugin."""
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["MESH_SHAPE"] = "1,2"
    os.environ["LOGGER_LEVEL"] = "DEBUG"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"

    os.environ["XLA_FLAGS"] = (
        "--xla_dump_to=/tmp/xla_dumps " "--xla_dump_hlo_as_dot " "--xla_dump_hlo_as_html " "--xla_dump_hlo_as_text"
    )

    xr.set_device_type("TT")
    xr.use_spmd()


def main():

    config: ExperimentConfig = generate_config(
        ExperimentConfig, "blacksmith/experiments/torch/mnist/test_mnist_training.yaml"
    )

    wandb_run = wandb.init(
        mode="online",
        project="mnist_dp_training",
        name="mnist_dp_training",
        tags=["wandb", "data_parallel"],
        dir="./wandb",
    )
    # Setup TT environment
    setup_tt_environment()

    # Device
    device = torch_xla.device()

    # Define mesh for multi-device training
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    axis_names = ("data", "model")
    mesh = Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)

    # Model
    model = MNISTLinear(
        config.net_config.input_size,
        config.net_config.hidden_size,
        config.net_config.output_size,
        bias=config.net_config.bias,
    )
    model = model.to(device)

    # Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_size = int(config.training_config.train_ratio * len(mnist_dataset))
    val_size = len(mnist_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(mnist_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.training_config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training_config.batch_size, shuffle=False, drop_last=True)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training_config.lr)

    # Training
    for epoch in range(config.training_config.epochs):
        model.train()
        steps = 0
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Reshape inputs and targets to (batch_size, -1)
            inputs = inputs.view(inputs.size(0), -1)
            targets = F.one_hot(targets, num_classes=10)
            targets = targets.view(targets.size(0), -1)

            inputs = inputs.to(device)
            targets = targets.to(device)

            xs.mark_sharding(inputs, mesh, ("data", None))
            xs.mark_sharding(targets, mesh, ("data", None))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = (outputs - targets).pow(2)
            loss = loss.mean(dim=1, keepdim=True)
            loss = loss.mean(dim=0, keepdim=True)
            loss.backward()
            xm.optimizer_step(optimizer, barrier=True)
            running_loss += loss.item()

            # Log loss every 100 steps
            if steps % 100 == 0:
                print(f"Step {steps}, Loss: {loss.item():.4f}")
            steps += 1

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        wandb_run.log({"train_loss": avg_loss, "epoch": epoch})

        # Validation, measure the accuracy
        if epoch % 5 == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.view(inputs.size(0), -1)
                    targets = F.one_hot(targets, num_classes=10)
                    targets = targets.view(targets.size(0), -1)

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)
                    pred = torch.argmax(outputs, dim=1)
                    label = torch.argmax(targets, dim=1)
                    correct += (pred == label).sum().item()
                    total += targets.size(0)

            print(f"Epoch {epoch}, Val Accuracy: {correct / total:.4f}")
            wandb_run.log({"val_accuracy": correct / total, "epoch": epoch})

    wandb_run.finish()


if __name__ == "__main__":
    main()
