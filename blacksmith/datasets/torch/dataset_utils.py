# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum

from blacksmith.datasets.torch.sst2.sst2_dataset import SSTDataset
from blacksmith.tools.configs import TrainingConfig


class AvailableDataset(Enum):
    SST2 = "sst2"


def get_dataset(config: TrainingConfig, split: str = "train", collate_fn=None):
    """Factory function to get the appropriate dataset based on the config."""
    dataset_id = config.dataset_id.lower()

    if dataset_id == AvailableDataset.SST2.value:
        return SSTDataset(config, split, collate_fn=collate_fn)

    available_datasets = [ds.value for ds in AvailableDataset]
    raise ValueError(f"Unsupported dataset: {dataset_id}. Available options are: {available_datasets}")
