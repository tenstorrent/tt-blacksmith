# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torch_geometric.datasets import Planetoid

# Map config.dataset_id -> Planetoid citation-network name.
PLANETOID_NAMES = {
    "pubmed": "PubMed",
    "cora": "Cora",
    "citeseer": "CiteSeer",
}

DATA_ROOT = "data"


def load_dataset(config, logger):
    """Load a Planetoid citation-network dataset selected by config.dataset_id."""
    name = PLANETOID_NAMES.get(config.dataset_id.lower())
    if name is None:
        raise ValueError(
            f"Unsupported dataset_id: {config.dataset_id!r}. Available options are: {sorted(PLANETOID_NAMES)}"
        )

    dataset = Planetoid(root=DATA_ROOT, name=name)
    data = dataset[0]
    logger.info(f"Loaded {name} dataset:")
    logger.info(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    logger.info(f"  Features: {data.num_node_features}, Classes: {dataset.num_classes}")
    logger.info(f"  Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
    return data
