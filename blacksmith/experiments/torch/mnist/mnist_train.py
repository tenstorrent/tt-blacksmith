# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torchvision
from torchvision import transforms
import torch_xla.core.xla_model as xm
from torch_xla.experimental import plugins
from torch.utils.data import DataLoader

from blacksmith.models.torch.mnist.mnist_linear import MNISTLinear
from blacksmith.datasets.torch.mnist.dataloader import load_mnist_torch
import torch_xla
import wandb

# --------------------------------
# Plugin registration
# --------------------------------
os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"



class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            return "/localdev/abogdanovic/tt-xla/build/src/tt/pjrt_plugin_tt.so"


plugins.register_plugin("TT", TTPjrtPlugin())
torch_xla.sync(True)


# --------------------------------
# Test run
# --------------------------------
def mnist():
    # Instantiate model.
    model: torch.nn.Module = MNISTCNNDropoutModel().to(dtype=torch.bfloat16)

    # Put it in inference mode and compile it.
    model = model.eval()
    model.compile(backend="openxla")

    # Generate inputs.
    input = torch.ones((4, 1, 28, 28), dtype=torch.bfloat16)

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = input.to(device)
    model = model.to(device)

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        output = model(input)

    print(output)


# --------------------------------
# Training function
# --------------------------------

def mnist_train():

    # Enable online mode for wandb
    run = wandb.init(mode="online", project="mnist_training", name="mnist_training")

    # Instantiate model.
    print("Initializing MNIST model...")
    model: torch.nn.Module = MNISTLinear(input_size=784, hidden_size=512, output_size=10).to(dtype=torch.bfloat16)

    # Define transformations for the dataset.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset. 
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

    # Define loss function and optimizer.
    print("Define loss function and optimizer...")
    loss_fn = torch.nn.MSELoss()

    # Connect the device.
    device = xm.xla_device()
    print(f"Using device: {device}")

    # Move model to device.
    model = model.to(device)

    # Create optimizer 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    xm.optimizer_step(optimizer)

    # Training loop.
    print("Training loop...")
    model.train()
    num_epochs = 5
    for epoch in range(num_epochs):  
        sum_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.view(inputs.size(0), -1)  
            targets = targets.view(targets.size(0), -1) 
            inputs, targets = inputs.to(device, dtype=torch.bfloat16), targets.to(device, dtype=torch.bfloat16)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            # Run optimizer step on host.
            xm.optimizer_step(optimizer)
            torch_xla.sync(True)

# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # Run MNIST inference or training function
    # mnist()
    mnist_train()
