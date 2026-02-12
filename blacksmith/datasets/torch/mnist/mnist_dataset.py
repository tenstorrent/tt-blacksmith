# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST as mnist_dataset

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig

# Constants for MNIST
MEAN = 0.1307
STD = 0.3081
NUM_CLASSES = 10


class MNISTDataset(BaseDataset):
    def __init__(self, config: TrainingConfig, split="train", collate_fn=None):
        self.config = config
        self.split = split
        self.collate_fn = collate_fn

        self._prepare_dataset()

    def _prepare_dataset(self):
        dtype = eval(self.config.dtype)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((MEAN,), (STD,)),  # Mean and std for MNIST
                transforms.Lambda(lambda x: x.view(-1)),  # Flatten image
                transforms.Lambda(lambda x: x.to(dtype)),  # Convert to dtype
            ],
        )
        target_transform = transforms.Compose(
            [
                transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long)),
                transforms.Lambda(lambda y: F.one_hot(y, num_classes=NUM_CLASSES).to(dtype)),
            ]
        )

        self.dataset = mnist_dataset(
            root="data",
            train=self.split == "train",
            download=True,
            transform=transform,
            target_transform=target_transform,
        )

    def __getitem__(self, idx: int):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def _get_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset, batch_size=self.config.batch_size, shuffle=self.split == "train", drop_last=True
        )
