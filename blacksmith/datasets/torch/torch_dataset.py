# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from itertools import islice
from typing import Dict, Union

from torch.utils.data import DataLoader, Dataset

from blacksmith.tools.templates.configs import TrainingConfig


class TestDataLoaderWrapper:
    """
    Wrapper for DataLoader that applies itertools.islice to limit steps per epoch.
    Used for fast testing by limiting the number of batches processed.
    """

    def __init__(self, dataloader: DataLoader, max_steps_per_epoch: int):
        """
        Args:
            dataloader: The DataLoader to wrap
            max_steps_per_epoch: Maximum number of batches to yield per epoch
        """
        self.dataloader = dataloader
        self.max_steps_per_epoch = max_steps_per_epoch

    def __iter__(self):
        """Return an iterator limited by max_steps_per_epoch."""
        return islice(iter(self.dataloader), self.max_steps_per_epoch)

    def __len__(self):
        """Return the limited length."""
        original_len = len(self.dataloader)
        return min(original_len, self.max_steps_per_epoch)

    def __getattr__(self, name):
        """Delegate attribute access to the wrapped dataloader."""
        return getattr(self.dataloader, name)


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

    @abstractmethod
    def _get_dataloader(self) -> DataLoader:
        """Create and return a DataLoader for this dataset"""
        raise NotImplementedError("Please Implement this method in the subclass")

    def __len__(self) -> int:
        """Return the number of examples in the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single example from the dataset"""
        pass

    def _prepare_test_dataloader(self, dataloader: DataLoader) -> Union[DataLoader, TestDataLoaderWrapper]:
        """
        Prepare a dataloader for test runs.

        If `self.config.test_config.max_steps_per_epoch` is set, returns a
        `TestDataLoaderWrapper` that yields at most that many batches per epoch.
        Otherwise, returns the original dataloader unchanged.

        Args:
            dataloader: The DataLoader to optionally wrap.

        Returns:
            The original `dataloader`, or a `TestDataLoaderWrapper` that limits the
            number of batches yielded per epoch.
        """
        if hasattr(self.config, "test_config") and self.config.test_config:
            max_steps = self.config.test_config.max_steps_per_epoch
            if max_steps is not None:
                return TestDataLoaderWrapper(dataloader, max_steps)
        return dataloader

    def get_dataloader(self) -> DataLoader:
        return self._prepare_test_dataloader(self._get_dataloader())
