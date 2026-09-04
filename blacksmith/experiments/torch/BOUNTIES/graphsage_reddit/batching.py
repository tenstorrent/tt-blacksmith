# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.data import Data


@dataclass(frozen=True)
class PreparedNeighborBatch:
    """A NeighborLoader batch ready for a CPU or TT training step."""

    x: torch.Tensor
    edge_index: torch.Tensor
    target_y: torch.Tensor
    target_mask: torch.Tensor
    target_capacity: int
    target_count: int


def sampled_graph_capacity(batch_size: int, num_neighbors: list[int]) -> tuple[int, int]:
    """Return conservative fixed node and edge capacities for neighbor sampling.

    The bound assumes that every sampled neighbor is unique and that a hop may
    expand every node discovered so far. One extra node is reserved as an
    isolated sentinel for padded edges.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if any(fanout < 0 for fanout in num_neighbors):
        raise ValueError("static shapes require finite non-negative fanouts")

    nodes = batch_size
    edges = 0
    for fanout in num_neighbors:
        sampled_edges = nodes * fanout
        edges += sampled_edges
        nodes += sampled_edges

    return nodes + 1, max(edges, 1)


def prepare_neighbor_batch(
    batch: Data,
    device: torch.device,
    seed_capacity: int,
    num_neighbors: list[int],
    static_shapes: bool,
) -> PreparedNeighborBatch:
    """Move a sampled graph to ``device`` and optionally pad every tensor.

    NeighborLoader puts the seed nodes first. With static shapes enabled, seed
    labels are padded and masked, graph nodes are zero-padded, and unused edges
    become self-edges on a reserved sentinel node. The sentinel is disconnected
    from the real graph, so padding cannot change real-node outputs.
    """
    target_count = int(batch.batch_size)
    if target_count > seed_capacity:
        raise ValueError(f"batch has {target_count} seed nodes, exceeding capacity {seed_capacity}")

    x = batch.x
    edge_index = batch.edge_index
    target_y = batch.y[:target_count]

    if static_shapes:
        node_capacity, edge_capacity = sampled_graph_capacity(seed_capacity, num_neighbors)
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        if num_nodes >= node_capacity:
            raise ValueError(
                f"sampled graph has {num_nodes} nodes; expected fewer than "
                f"the sentinel-inclusive capacity {node_capacity}"
            )
        if num_edges > edge_capacity:
            raise ValueError(f"sampled graph has {num_edges} edges, exceeding capacity " f"{edge_capacity}")

        sentinel = node_capacity - 1
        x = F.pad(x, (0, 0, 0, node_capacity - num_nodes))
        padded_edges = torch.full(
            (2, edge_capacity - num_edges),
            sentinel,
            dtype=edge_index.dtype,
            device=edge_index.device,
        )
        edge_index = torch.cat((edge_index, padded_edges), dim=1)
        target_y = F.pad(target_y, (0, seed_capacity - target_count))
        target_capacity = seed_capacity
    else:
        target_capacity = target_count

    target_mask = torch.arange(target_capacity, device=target_y.device) < target_count
    return PreparedNeighborBatch(
        x=x.to(device),
        edge_index=edge_index.to(device),
        target_y=target_y.to(device),
        target_mask=target_mask.to(device),
        target_capacity=target_capacity,
        target_count=target_count,
    )


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute cross entropy over a static target tensor using a mask."""
    losses = F.cross_entropy(logits, targets, reduction="none")
    weights = mask.to(losses.dtype)
    return (losses * weights).sum() / weights.sum()


def masked_correct(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Count correct predictions selected by ``mask`` without dynamic indexing."""
    correct = (logits.argmax(dim=1) == targets).to(logits.dtype)
    return (correct * mask.to(logits.dtype)).sum()
