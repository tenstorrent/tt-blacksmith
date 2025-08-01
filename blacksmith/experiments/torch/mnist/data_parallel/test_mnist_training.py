# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import copy
import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import torch_xla.distributed.parallel_loader as pl
from blacksmith.tools.cli import generate_config
from blacksmith.datasets.torch.mnist.dataloader import load_mnist_torch
from blacksmith.models.torch.mnist.mnist_linear import MNISTLinear
from blacksmith.experiments.torch.mnist.configs import ExperimentConfig
from blacksmith.tools.torch_xla_utils import init_device
import os

import wandb



os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"

# --------------------------------
# Load device configuration
# --------------------------------

config: ExperimentConfig = generate_config(
    ExperimentConfig, "blacksmith/experiments/torch/mnist/test_mnist_training.yaml"
)
# torch_xla.sync(wait=True)

def setup_tt_environment():
    """Setup TensorTrent environment and plugin."""
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["MESH_SHAPE"] = "2,4"
    os.environ["LOGGER_LEVEL"] = "DEBUG"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    from torch_xla.experimental import plugins
    # TODO: Replace with init device when available

    class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            return os.path.join(
                os.path.dirname(__file__),
                "/localdev/abogdanovic/tt-xla/build/src/tt/pjrt_plugin_tt.so",
            )

    plugins.register_plugin("TT", TTPjrtPlugin())
    # init_device()
    xr.use_spmd()
    torch_xla.sync(wait=True)


def training_on_multiple_devices():
    logger_config = config.logger_config

    wandb_run = wandb.init(
        mode="online",
        project="mnist_dp_training",
        name="mnist_dp_training",
        tags=["wandb"],
        dir=logger_config.wandb_dir,
    )

    if logger_config.log_hyperparameters:
        wandb_run.config.update(config.model_dump())

    num_steps = 32
    batch_size = 2
    setup_tt_environment()
    torch.manual_seed(1)

    # Model
    model = MNISTLinear(784, 512,10, bias=True).to(torch.bfloat16)
    model = model.train()

    inputs_and_targets = []
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    data_iterator = iter(dataloader)
    inputs = []
    targets = []
    for _ in range(num_steps):
        test_input, target = next(data_iterator)
        print(f"Test input shape: {test_input.shape}, Target shape: {target.shape}")
        test_input = test_input.to(torch.bfloat16)
        inputs.append(test_input)
        targets.append(target)
        inputs_and_targets.append((test_input, target))

    # xr.use_spmd(True)
    num_devices = xr.global_runtime_device_count()
    print(f"Running on {num_devices} devices")
    mesh_shape = (num_devices, 1, 1, 1)
    axis_names = ("data", "c", "h", "w")
    device_ids = np.arange(num_devices).reshape(mesh_shape)
    mesh = Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)
    print(f"Mesh shape: {mesh_shape}, Device IDs: {device_ids}")

    inputs = torch.cat(inputs, dim=0)
    targets = torch.cat(targets, dim=0)
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=num_steps * batch_size, shuffle=True)

    input_sharding = xs.ShardingSpec(mesh, ("data", None, None, None))
    train_device_loader = pl.MpDeviceLoader(
        loader,
        torch_xla.device(),
        input_sharding=input_sharding,
    )
    # Device
    device = torch_xla.device()
    model = model.to(device)

    # Optimizer and Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.NLLLoss()

    # Training
    for epoch in range(20):  # Using 2 epochs for demonstration
        model.train()
        train_loss = 0.0
        for step, (inputs, targets) in enumerate(train_device_loader):
            inputs = inputs.view(inputs.size(0), -1)

            inputs = inputs.to(device, dtype=torch.bfloat16)
            targets = targets.to(device)

            # Mark sharding for targets
            xs.mark_sharding(targets, mesh, ("data",))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            xm.optimizer_step(optimizer)
            torch_xla.sync(wait=True)
            train_loss += loss.cpu().item()

        avg_train_loss = train_loss / len(train_device_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1})

    print("Training complete. Saving model parameters.")
    
 
def test_mnist_ttxla():
    training_on_multiple_devices()