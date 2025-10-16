# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Dict

from torch.utils.data import DataLoader, Dataset

from blacksmith.experiments.torch.llama.configs import TrainingConfig


class BaseDataset(Dataset, ABC):
    """Abstract base class for all PyTorch dataset implementations"""

    def __init__(self, config: TrainingConfig, split: str = "train"):
        """
        Args:
            config: Training configuration
            split: Dataset split to use ("train", "validation", "test", etc.)
        """
        self.config = config
        self.split = split

        self._prepare_dataset(split)

    @abstractmethod
    def _prepare_dataset(self, split: str):
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
