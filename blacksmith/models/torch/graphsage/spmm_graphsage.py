# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Scatter-free GraphSAGE mean aggregation for Tenstorrent devices."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv

SEGSUM_BLOCK_SIZE = 16384


def _segment_sum(
    values: Tensor,
    group_index: Tensor,
    num_groups: int,
    block_size: int = SEGSUM_BLOCK_SIZE,
) -> Tensor:
    """Sum edge values by destination using one-hot matrix multiplies.

    ``torch_geometric`` implements this operation with scatter reductions. On TT,
    those reductions produce large serialized graphs. The one-hot matrix for each
    edge block is transient, which bounds memory while keeping both the forward and
    input-gradient paths scatter-free. XLA matmuls use bf16 inputs with an fp32
    accumulator; CPU keeps the input dtype for exact PyG parity checks.
    """
    feature_count = values.size(1)
    edge_count = values.size(0)
    node_ids = torch.arange(num_groups, device=values.device).unsqueeze(1)
    matmul_dtype = torch.bfloat16 if values.device.type == "xla" else values.dtype
    accumulation_dtype = torch.float32 if values.device.type == "xla" else values.dtype
    output = torch.zeros(
        num_groups,
        feature_count,
        dtype=accumulation_dtype,
        device=values.device,
    )

    for start in range(0, edge_count, block_size):
        end = min(start + block_size, edge_count)
        one_hot = (node_ids == group_index[start:end].unsqueeze(0)).to(matmul_dtype)
        block_sum = one_hot @ values[start:end].to(matmul_dtype)
        output = output + block_sum.to(accumulation_dtype)

    return output.to(values.dtype)


class _SegmentSum(torch.autograd.Function):
    """Edge-to-node sum with a gather-only backward."""

    @staticmethod
    def forward(ctx: Any, values: Tensor, group_index: Tensor, num_groups: int) -> Tensor:
        ctx.save_for_backward(group_index)
        return _segment_sum(values, group_index, num_groups)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None, None]:
        (group_index,) = ctx.saved_tensors
        return grad_output.index_select(0, group_index), None, None


class _EdgeGather(torch.autograd.Function):
    """Node-to-edge gather whose input-gradient avoids scatter-add."""

    @staticmethod
    def forward(ctx: Any, node_features: Tensor, node_index: Tensor, num_nodes: int) -> Tensor:
        ctx.save_for_backward(node_index)
        ctx.num_nodes = num_nodes
        return node_features.index_select(0, node_index)

    @staticmethod
    def backward(ctx: Any, grad_edges: Tensor) -> tuple[Tensor, None, None]:
        (node_index,) = ctx.saved_tensors
        return _segment_sum(grad_edges, node_index, ctx.num_nodes), None, None


class SpMMGraphSAGEConv(SAGEConv):
    """Default homogeneous :class:`SAGEConv` implemented without scatter.

    The layer preserves the parameter layout and state-dict keys of PyG's default
    ``SAGEConv``. For every target node it computes the duplicate-aware mean of
    incoming source features, then applies ``lin_l`` and the root transform
    ``lin_r``. Targets with no incoming edges receive a zero aggregate, matching
    PyG's mean aggregation behavior.

    This implementation intentionally supports the default homogeneous GraphSAGE
    mode used by this experiment: mean aggregation, root weights, no pre-projection,
    and no output normalization.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            in_channels,
            out_channels,
            aggr="mean",
            normalize=False,
            root_weight=True,
            project=False,
            bias=True,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        size: tuple[int, int] | None = None,
    ) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("SpMMGraphSAGEConv supports homogeneous tensor inputs only")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if size is not None and tuple(size) != (x.size(0), x.size(0)):
            raise ValueError("SpMMGraphSAGEConv supports homogeneous graphs only")

        num_nodes = x.size(0)
        source = edge_index[0]
        target = edge_index[1]

        messages = _EdgeGather.apply(x, source, num_nodes)
        neighbor_sum = _SegmentSum.apply(messages, target, num_nodes)

        edge_ones = x.new_ones((edge_index.size(1), 1))
        degree = _SegmentSum.apply(edge_ones, target, num_nodes)
        neighbor_mean = neighbor_sum / degree.clamp_min(1)

        output = self.lin_l(neighbor_mean) + self.lin_r(x)
        if self.normalize:
            output = F.normalize(output, p=2.0, dim=-1)
        return output
