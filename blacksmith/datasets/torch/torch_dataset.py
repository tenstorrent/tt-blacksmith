# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict

from torch.utils.data import DataLoader, Dataset

from blacksmith.datasets.torch.nerf.blender import BlenderDataset
from blacksmith.datasets.torch.banking77.banking77_dataset import Banking77Dataset
from blacksmith.datasets.torch.mnist.mnist_dataset import MNISTDataset
from blacksmith.datasets.torch.text2sql.text2sql_dataset import TextToSQLDataset
from blacksmith.datasets.torch.sst2.sst2_dataset import SSTDataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig


class AvailableDataset(Enum):
    MNIST = "mnist"
    NERF = "nerf"
    SST2 = "sst2"
    TEXT2SQL = "text2sql"
    BANKING77 = "banking77"


class BaseDataset(Dataset, ABC):
    """Abstract base class for all PyTorch dataset implementations"""

    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        """
        Args:
            config: Training configuration
            split: Dataset split to use ("train", "validation", "test", etc.)
            collate_fn: Function to collate samples into batches
        """
        self.config = config
        self.split = split
        self.collate_fn = collate_fn

        self._prepare_dataset()

    @abstractmethod
    def _prepare_dataset(self):
        """Load and prepare the dataset"""
        pass

    def __len__(self) -> int:
        """Return the number of examples in the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single example from the dataset"""
        pass

    @abstractmethod
    def get_dataloader(self) -> DataLoader:
        """Create and return a DataLoader for this dataset"""
        pass


def get_dataset(config: TrainingConfig, split: str = "train", collate_fn=None):
    """Factory function to get the appropriate dataset based on the config"""
    dataset_id = config.get("dataset_id", "").lower()

    if dataset_id == AvailableDataset.MNIST.value:
        return MNISTDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.NERF.value:
        return BlenderDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.SST2.value:
        return SSTDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.TEXT2SQL.value:
        return TextToSQLDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.BANKING77.value:
        return Banking77Dataset(config, split, collate_fn=collate_fn)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_id}")
