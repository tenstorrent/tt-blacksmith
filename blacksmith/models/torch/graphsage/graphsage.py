# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv

from blacksmith.models.torch.graphsage.spmm_graphsage import SpMMGraphSAGEConv


class GraphSAGE(torch.nn.Module):
    """Two-layer GraphSAGE with stock or scatter-free mean aggregation."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float,
        use_spmm: bool = False,
    ) -> None:
        super().__init__()
        conv_class = SpMMGraphSAGEConv if use_spmm else SAGEConv
        self.conv1 = conv_class(in_channels, hidden_channels)
        self.conv2 = conv_class(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
