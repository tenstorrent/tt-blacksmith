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

from torch.utils.tensorboard import SummaryWriter
import wandb

# --------------------------------
# Plugin registration
# --------------------------------
os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"



class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            return os.path.join(
                os.path.dirname(__file__), "/localdev/abogdanovic/tt-xla/build/src/tt/pjrt_plugin_tt.so"
            )


plugins.register_plugin("TT", TTPjrtPlugin())


# --------------------------------
# Test run
# --------------------------------
def mnist():
    # Instantiate model.
    model: torch.nn.Module = MNISTLinear(input_size=784, hidden_size=512, output_size=10).to(dtype=torch.bfloat16)

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
    # Enable live sync with TensorBoard
    wandb.init(project ="mnist_training", sync_tensorboard=True)

    # Create a SummaryWriter for TensorBoard logs
    writer = SummaryWriter(log_dir="runs")

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
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

    # Define loss function and optimizer.
    print("Define loss function and optimizer...")
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

    # Connect the device.
    device = xm.xla_device()
    print(f"Using device: {device}")

    # Move model to device.
    model = model.to(device)

    # Training loop.
    print("Training loop...")
    model.train()
    for epoch in range(20):  
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.view(inputs.size(0), -1)  
            inputs, targets = inputs.to(device, dtype=torch.bfloat16), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/5], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                # Log loss to TensorBoard
                writer.add_scalar("loss", loss.item(), epoch * len(train_loader) + batch_idx)

        print(f"Epoch [{epoch+1}/5] completed.")
    
    # Close the TensorBoard writer
    writer.close()
    # Sync with wandb
    wandb.finish()

# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # Uncomment the desired function to run.
    # mnist()
    mnist_train()
