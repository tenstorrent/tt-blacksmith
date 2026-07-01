# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from blacksmith.models.torch.gatv2_pubmed.spmm_gatv2 import SpMMGATv2Conv, setup_graph


class GATv2(nn.Module):
    """2-layer GATv2 for PubMed node classification.

    With ``use_spmm=True`` the scatter-based ``GATv2Conv`` is swapped for the
    SpMM (matmul) ``SpMMGATv2Conv`` so the model trains natively on TT without
    hitting the scatter tile-padding OOM (tt-mlir#8887; the earlier serial-chain
    blowup tt-mlir#8714 was fixed by #8718); the math is unchanged.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6, use_spmm=False):
        super().__init__()
        self.dropout = dropout
        self.use_spmm = use_spmm
        conv_cls = SpMMGATv2Conv if use_spmm else GATv2Conv
        self.conv1 = conv_cls(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = conv_cls(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        if self.use_spmm:
            # Bind the (self-looped) graph to the SpMM convs; full-batch, so the graph is
            # identical every step. Cheap: stores index views + builds the tiny M matrix.
            setup_graph((self.conv1, self.conv2), edge_index, x.size(0))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def get_model(config, num_features, num_classes, device, logger):
    """Instantiate GATv2 model and move to device."""
    model = GATv2(
        in_channels=num_features,
        hidden_channels=config.hidden_channels,
        out_channels=num_classes,
        heads=config.heads,
        dropout=config.dropout,
        use_spmm=config.use_spmm,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params}")
    logger.info(f"Trainable parameters: {trainable_params}")
    return model
