# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum

from blacksmith.configs import TrainerConfig
from blacksmith.datasets.alpaca import AlpacaDataset
from blacksmith.datasets.custom import CustomLLMDataset
from blacksmith.datasets.sst2 import SSTDataset


class AvailableDataset(Enum):
    SST2 = "sst2"
    ALPACA = "alpaca"
    CUSTOM = "custom"


def get_dataset(config: TrainerConfig, split: str = "train", collate_fn=None):
    """Factory function to get the appropriate dataset based on the config"""
    dataset_id = config.dataset_id.lower()

    if dataset_id == AvailableDataset.SST2.value:
        return SSTDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.ALPACA.value:
        return AlpacaDataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.CUSTOM.value:
        return CustomLLMDataset(config, split, collate_fn=collate_fn)
    else:
        available_datasets = [ds.value for ds in AvailableDataset]
        raise ValueError(f"Unsupported dataset: {dataset_id}. Available options are: {available_datasets}")
