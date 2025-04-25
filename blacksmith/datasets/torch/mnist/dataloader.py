# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST as mnist_dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch


def load_mnist_torch(dtype, batch_size):
    dtype = dtype
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std for MNIST
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten image
            transforms.Lambda(lambda x: x.to(dtype)),  # Convert to dtype
        ],
    )
    target_transform = transforms.Compose(
        [
            transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long)),
            transforms.Lambda(lambda y: F.one_hot(y, num_classes=10).to(dtype)),
        ]
    )

    train_dataset = mnist_dataset(
        root="data",
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    test_dataset = mnist_dataset(
        root="data",
        train=False,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )

    # Drop last to ensure all batches are the same size as compiled model expects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader
