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
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh, XLAShardedTensor, XLAShard
import torch_xla.distributed.parallel_loader as pl
from torch_xla.experimental import plugins
from blacksmith.tools.cli import generate_config
from blacksmith.models.torch.mnist.mnist_linear import MNISTLinear
from blacksmith.experiments.torch.mnist.configs import ExperimentConfig
from blacksmith.tools.torch_xla_utils import init_device
import os
import wandb


# --------------------------------
# Load device configuration
# --------------------------------
# torch_xla.sync(wait=True)


def setup_tt_environment():
    """Setup TensorTrent environment and plugin."""
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    # os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["MESH_SHAPE"] = "2,1"
    os.environ["LOGGER_LEVEL"] = "DEBUG"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"

    # # TODO: Replace with init device when available

    # class TTPjrtPlugin(plugins.DevicePlugin):
    #     def library_path(self):
    #         return os.path.join(
    #             os.path.dirname(__file__),
    #             "/localdev/abogdanovic/tt-xla/build/src/tt/pjrt_plugin_tt.so",
    #         )

    # plugins.register_plugin("TT", TTPjrtPlugin())
    # init_device()
    xr.use_spmd()
    # torch_xla.sync(wait=True)


def get_loader(data_loader, num_steps, batch_size):
    data_iterator = iter(data_loader)
    inputs = []
    targets = []

    for _ in range(min(num_steps, len(data_loader))):
        input, target = next(data_iterator)
        inputs.append(input.to(torch.bfloat16))
        targets.append(target)

    inputs = torch.cat(inputs, dim=0)
    targets = torch.cat(targets, dim=0)
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return pl.MpDeviceLoader(loader, torch_xla.device())


def apply_tensor_parallel_sharding(model, mesh):
    # Apply tensor parallel sharding to model parameters (the first linear layer)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "weight"):
            if "linear_relu_stack.0" in name:
                xs.mark_sharding(module.weight[..., None], mesh, (None, "tensor", None, ))


def training_on_multiple_devices():

    config: ExperimentConfig = generate_config(
        ExperimentConfig, "blacksmith/experiments/torch/mnist/test_mnist_training.yaml"
    )

    # logger_config = config.logger_config

    # wandb_run = wandb.init(
    #     mode="online",
    #     project="mnist_tp_training",
    #     name="mnist_tp_training",
    #     tags=["wandb", "tensor_parallel"],
    #     dir=logger_config.wandb_dir,
    # )

    # if logger_config.log_hyperparameters:
    #     wandb_run.config.update(config.model_dump())

    num_steps = 32
    batch_size = 2
    setup_tt_environment()
    torch.manual_seed(1)

    # Setup tensor parallel mesh using XLAMesh
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1, 1, 1)
    axis_names = ("data", "tensor", "model", "batch")
    os.environ["MESH_SHAPE"] = f"{mesh_shape[0]},{mesh_shape[1]}"

    # Create device IDs array and Mesh
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)
    xs.set_global_mesh(mesh)

    # Model
    model = MNISTLinear(784, 512, 10, bias=False).to(torch.bfloat16)

    # Dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_size = int(0.8 * len(mnist_dataset))
    val_size = len(mnist_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(mnist_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Define input sharding for data parallelism using XLAMesh
    input_sharding = xs.ShardingSpec(mesh, ("data", None, None, None))

    # train_device_loader = get_loader(
    #     data_loader=train_dataloader, num_steps=num_steps, batch_size=batch_size
    # )
    # val_device_loader = get_loader(
    #     data_loader=val_dataloader, num_steps=num_steps, batch_size=batch_size
    # )

    train_device_loader = pl.MpDeviceLoader(
        train_dataloader,
        torch_xla.device(),
        input_sharding=input_sharding,
    )

    val_device_loader = pl.MpDeviceLoader(
        val_dataloader,
        torch_xla.device(),
        input_sharding=input_sharding,
    )

    # Device
    device = torch_xla.device()
    model = model.to("xla")

    print(f"Using device: {device}, mesh shape: {mesh.mesh_shape}, axis names: {mesh.axis_names}")
    # Apply tensor parallel sharding to model parameters
    apply_tensor_parallel_sharding(model, mesh)

    # Optimizer and Loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Training
    print("Training start")
    model.train()
    for epoch in range(1):
        train_loss = 0.0

        for step, (inputs, targets) in enumerate(train_device_loader):
            inputs = inputs.view(inputs.size(0), -1)
            # targets = targets.view(targets.size(0), -1)

            inputs = inputs.to("xla", dtype=torch.bfloat16)
            targets = targets.to("xla", dtype=torch.int32)

            optimizer.zero_grad()
            outputs = model(inputs)
            print(f"Outputs: {outputs.shape}")
            print(f"Targets: {targets.shape}")
            loss = loss_fn(outputs, targets)
            loss.backward()
            # train_loss += loss.cpu().item()
            # import pdb; pdb.set_trace()
            xm.optimizer_step(optimizer)
            # torch_xla.sync(wait=True)
            train_loss +=  0 # loss.cpu().item()
            # print(f"epoch {step + 1}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_device_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")
        # wandb.log({"train_loss": avg_train_loss, "epoch": epoch + 1})

        # # Validation
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for step, (val_inputs, val_targets) in enumerate(val_device_loader):
        #         val_inputs = val_inputs.view(val_inputs.size(0), -1)

        #         val_inputs = val_inputs.to(device, dtype=torch.bfloat16)
        #         val_targets = val_targets.to(device)

        #         outputs = model(val_inputs)
        #         loss = loss_fn(outputs)
        #         # val_loss += loss.item()

        # avg_val_loss = val_loss / len(val_device_loader)
        # print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # # wandb.log({"val_loss": avg_val_loss}, step=epoch + 1)

    print("Training complete. Saving model parameters.")



if __name__ == "__main__":
    xr.set_device_type("TT")
    training_on_multiple_devices()
