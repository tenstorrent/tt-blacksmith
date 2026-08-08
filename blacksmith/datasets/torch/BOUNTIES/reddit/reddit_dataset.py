# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader

from blacksmith.datasets.torch.torch_dataset import BaseDataset, TestDataLoaderWrapper
from blacksmith.tools.templates.configs import TrainingConfig


class RedditDataset(BaseDataset):
    def __init__(self, config: TrainingConfig, split: str = "train") -> None:
        self._loaders: dict[str, NeighborLoader] = {}
        super().__init__(config=config, split=split)

    def _prepare_dataset(self) -> None:
        self.dataset = Reddit(root=self.config.dataset_root)
        self.data = self.dataset[0]

    def _get_dataloader(self) -> NeighborLoader:
        return self._get_neighbour_loader(self.split)

    def _get_neighbour_loader(self, split: str) -> NeighborLoader:
        masks = {
            "train": self.data.train_mask,
            "val": self.data.val_mask,
            "test": self.data.test_mask,
        }
        if split not in masks:
            valid_splits = ", ".join(masks)
            raise ValueError(f"Unknown Reddit split '{split}'. Expected one of: {valid_splits}.")

        if split in self._loaders:
            return self._loaders[split]

        batch_size = self.config.batch_size if split == "train" else self.config.val_batch_size
        self._loaders[split] = NeighborLoader(
            self.data,
            num_neighbors=self.config.num_neighbors,
            batch_size=batch_size,
            input_nodes=masks[split],
            shuffle=(split == "train"),
            subgraph_type="directional",
        )
        return self._loaders[split]

    def get_neighbour_loader(self, split: str | None = None) -> NeighborLoader | TestDataLoaderWrapper:
        requested_split = self.split if split is None else split
        loader = self._get_neighbour_loader(requested_split)
        # Applies the shared CI test-mode step limit (config.test_config.max_steps_per_epoch)
        # to every split, since BaseDataset.get_dataloader() is bypassed here.
        return self._prepare_test_dataloader(loader)

    @property
    def num_features(self) -> int:
        return self.dataset.num_features

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    @property
    def num_nodes(self) -> int:
        return self.data.num_nodes

    @property
    def num_edges(self) -> int:
        return self.data.num_edges

    @property
    def train_nodes(self) -> int:
        return int(self.data.train_mask.sum())

    @property
    def val_nodes(self) -> int:
        return int(self.data.val_mask.sum())

    @property
    def test_nodes(self) -> int:
        return int(self.data.test_mask.sum())
