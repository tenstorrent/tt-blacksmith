# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Any, Dict

from blacksmith.configs import TrainerConfig


class BaseDataset(ABC):
    """Abstract base class for all dataset implementations.

    Framework-free: it only needs ``__len__`` / ``__getitem__``, which is all a torch
    ``DataLoader`` (or a ttml dataloader) asks of a dataset.
    """

    def __init__(self, config: TrainerConfig, split: str = "train", collate_fn=None):
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

    @abstractmethod
    def _get_dataloader(self) -> Any:
        """Create and return a DataLoader for this dataset"""
        raise NotImplementedError("Please Implement this method in the subclass")

    def __len__(self) -> int:
        """Return the number of examples in the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single example from the dataset"""
        pass

    def get_dataloader(self) -> Any:
        return self._get_dataloader()
