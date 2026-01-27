# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum

from blacksmith.datasets.torch.banking77.banking77_dataset import Banking77Dataset
from blacksmith.datasets.torch.mnist.mnist_dataset import MNISTDataset
from blacksmith.datasets.torch.nerf.blender import BlenderDataset
from blacksmith.datasets.torch.squadV2.squadV2_dataset import SquadV2Dataset
from blacksmith.datasets.torch.sst2.sst2_dataset import SSTDataset
from blacksmith.datasets.torch.stanfordcars.stanfordcars_dataset import (
    StanfordCarsDataset,
)
from blacksmith.datasets.torch.text2sql.text2sql_dataset import TextToSQLDataset
from blacksmith.tools.templates.configs import TrainingConfig


class AvailableDataset(Enum):
    MNIST = "mnist"
    NERF = "nerf"
    SST2 = "sst2"
    TEXT2SQL = "text2sql"
    BANKING77 = "banking77"
    SQUADV2 = "squadv2"
    STANFORDCARS = "stanfordcars"


def get_dataset(config: TrainingConfig, split: str = "train", collate_fn=None):
    """Factory function to get the appropriate dataset based on the config"""
    dataset_id = config.dataset_id.lower()

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
    elif dataset_id == AvailableDataset.SQUADV2.value:
        return SquadV2Dataset(config, split, collate_fn=collate_fn)
    elif dataset_id == AvailableDataset.STANFORDCARS.value:
        return StanfordCarsDataset(config, split)
    else:
        available_datasets = [ds.value for ds in AvailableDataset]
        raise ValueError(f"Unsupported dataset: {dataset_id}. Available options are: {available_datasets}")
