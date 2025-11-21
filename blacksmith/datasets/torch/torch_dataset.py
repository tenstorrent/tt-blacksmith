# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Dict

from torch.utils.data import DataLoader, Dataset

from blacksmith.tools.templates.configs import TrainingConfig


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
