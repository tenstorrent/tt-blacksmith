# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from pydantic import BaseModel
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST as mnist_dataset

from thomas.models.torch.dtypes import TorchDType


@dataclass
class DataLoadingConfig(BaseModel):
    batch_size: int
    dtype: TorchDType
    pre_shuffle: bool


def load_dataset(config: DataLoadingConfig):
    print(config)
    dtype = config.dtype
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Mean and std for MNIST
            transforms.Lambda(lambda x: x.view(-1)),  # Flatten image
            transforms.Lambda(lambda x: x.to(dtype)),  # Convert to dtype
        ]
    )

    train_dataset = mnist_dataset(root="data", train=True, download=True, transform=transform)

    test_dataset = mnist_dataset(root="data", train=False, download=True, transform=transform)

    # Drop last to ensure all batches are the same size as compiled model expects
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader
