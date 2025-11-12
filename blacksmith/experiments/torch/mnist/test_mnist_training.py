# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import wandb

from blacksmith.tools.cli import generate_config
from blacksmith.models.torch.mnist.mnist_linear import MNISTLinear
from blacksmith.experiments.torch.mnist.configs import ExperimentConfig
import os


# --------------------------------
# Training loop
# --------------------------------
def test_training():

    config: ExperimentConfig = generate_config(
        ExperimentConfig, "blacksmith/experiments/torch/mnist/test_mnist_training.yaml"
    )

    xr.runtime.set_device_type("TT")

    logger_config = config.logger_config

    wandb_run = wandb.init(
        mode="online",
        project=config.experiment_name,
        name=config.experiment_name,
        tags=config.tags,
        dir=logger_config.wandb_dir,
    )

    if logger_config.log_hyperparameters:
        wandb_run.config.update(config.model_dump())

    # Model
    model = MNISTLinear(**config.net_config.model_dump()).to(dtype=getattr(torch, config.data_loading_dtype))

    # Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_size = int(config.training_config.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.training_config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training_config.batch_size, shuffle=False, drop_last=False)

    # Device
    device = xm.xla_device()
    model = model.to(device)

    # Optimizer and Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training_config.lr)
    loss_fn = eval(config.loss)()

    # Training
    for epoch in range(config.training_config.epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.view(inputs.size(0), -1)
            targets = targets.view(targets.size(0), -1)

            inputs = inputs.to(device, dtype=torch.bfloat16)
            targets = targets.to(device, dtype=torch.bfloat16)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            train_loss += loss.cpu().item()

            xm.optimizer_step(optimizer)
            torch_xla.sync(wait=True)

        avg_train_loss = train_loss / len(train_loader)
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1})

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.view(inputs.size(0), -1)
                targets = targets.view(targets.size(0), -1)

                inputs = inputs.to(device, dtype=torch.bfloat16)
                targets = targets.to(device, dtype=torch.bfloat16)

                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        wandb.log({"val_loss": avg_val_loss}, step=epoch + 1)


if __name__ == "__main__":
    test_training()
