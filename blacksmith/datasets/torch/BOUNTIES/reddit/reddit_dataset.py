# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborLoader

from blacksmith.experiments.torch.BOUNTIES.graphsage_reddit.configs import (
    GraphSAGEConfig,
)


@dataclass
class RedditLoaders:
    train_loader: NeighborLoader
    val_loader: NeighborLoader
    test_loader: NeighborLoader
    num_features: int
    num_classes: int
    num_nodes: int
    num_edges: int
    train_nodes: int
    val_nodes: int
    test_nodes: int


def get_reddit_loaders(config: GraphSAGEConfig) -> RedditLoaders:
    dataset = Reddit(root=config.dataset_root)
    data = dataset[0]

    train_loader = NeighborLoader(
        data,
        num_neighbors=config.num_neighbors,
        batch_size=config.batch_size,
        input_nodes=data.train_mask,
        shuffle=True,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=config.num_neighbors,
        batch_size=config.val_batch_size,
        input_nodes=data.val_mask,
        shuffle=False,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=config.num_neighbors,
        batch_size=config.val_batch_size,
        input_nodes=data.test_mask,
        shuffle=False,
    )

    return RedditLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        num_nodes=data.num_nodes,
        num_edges=data.num_edges,
        train_nodes=int(data.train_mask.sum()),
        val_nodes=int(data.val_mask.sum()),
        test_nodes=int(data.test_mask.sum()),
    )
