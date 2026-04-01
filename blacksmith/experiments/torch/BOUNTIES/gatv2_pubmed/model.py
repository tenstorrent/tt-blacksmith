# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self._fallback_to_cpu = False

    def enable_cpu_fallback(self):
        """Move GATv2Conv layers to CPU for fallback execution.

        GATv2Conv uses scatter-based message passing that does not
        compile on TT-XLA (scatter_reduce_, scatter_add_). The
        backward pass of gather ops inside GATv2Conv also generates
        scatter_add_ which fails on TT. Additionally, mixing TT and
        CPU ops in a single forward pass causes gradient corruption
        at the device boundary (torch_xla issue). Therefore the
        entire forward pass must run on CPU.
        """
        self.conv1 = self.conv1.cpu()
        self.conv2 = self.conv2.cpu()
        self._fallback_to_cpu = True

    def forward(self, x, edge_index):
        if self._fallback_to_cpu:
            return self._forward_cpu(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def _forward_cpu(self, x, edge_index):
        """Full forward pass on CPU.

        Runs the entire forward on CPU to avoid:
        1. scatter_reduce_ compilation failure on TT-XLA
        2. Gradient corruption at TT<->CPU device boundaries
        """
        x_cpu = x.cpu()
        ei_cpu = edge_index.cpu()
        x_cpu = F.dropout(x_cpu, p=self.dropout, training=self.training)
        x_cpu = self.conv1(x_cpu, ei_cpu)
        x_cpu = F.elu(x_cpu)
        x_cpu = F.dropout(x_cpu, p=self.dropout, training=self.training)
        x_cpu = self.conv2(x_cpu, ei_cpu)
        return F.log_softmax(x_cpu, dim=1)
