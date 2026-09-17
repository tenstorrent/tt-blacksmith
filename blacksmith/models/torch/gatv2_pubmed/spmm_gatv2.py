# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Scatter-free GATv2 convolution for Tenstorrent.
#
# Stock ``GATv2Conv`` aggregates messages with ``scatter_add`` / ``scatter_reduce_``.
# On the current TT stack a single logical scatter over a length-L index lowers to a
# serial chain of ~L/256 ``ttnn.scatter`` ops, so on full PubMed (88k edges) the
# aggregation expands to ~190k chained ops and a single forward pass does not finish
# compiling (tt-mlir issue tenstorrent/tt-mlir#8714).
#
# ``SpMMGATv2Conv`` keeps the exact GATv2 math but rewrites every node<->edge operation
# (the x_i/x_j feature lookups, the attention-softmax denominator, and the message
# aggregation) as a *matmul against a one-hot incidence*, which lowers to ``ttnn.matmul``
# and emits ZERO scatter. It is bit-equivalent to ``GATv2Conv`` on CPU (loss identical,
# per-parameter gradient cosine 1.0) and trains full PubMed to ~78% test accuracy on a
# Wormhole N300, matching the CPU baseline.
#
# The five things that make it run on TT today:
#   1. SpMM aggregation (no scatter)                  -> dodges the #8714 tiling blowup.
#   2. ``att`` stored flat [1, H*C] (not [1, H, C])   -> avoids a tile-padded sub-32
#      reshape that ttnn rejects.
#   3. per-head channel reduction via a constant      -> avoids reshaping [E, H*C] to
#      block-ones matmul (``ConstMatmul``)               [E, H, C] (same reshape issue).
#   4. blocked one-hot built on the fly in bf16       -> the full [N, E] incidence would
#      (1.0/0.0 are exact in bf16; matmul still          OOM; one-hot bytes, not block
#      accumulates in fp32 on Wormhole)                  count, are the memory lever.
#   5. callers must use a static masked loss          -> boolean-mask indexing
#      (see masked_nll_loss / masked_accuracy)           (``out[mask]``) has a
#                                                         dynamic-shape backward that
#                                                         ttnn rejects.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops

SEGSUM_BLOCK = 16384  # edge-block size; bounds the transient one-hot to [N, SEGSUM_BLOCK]


def _segsum(vals, group_idx, num_nodes, block=SEGSUM_BLOCK):
    """Scatter-free segment sum: out[n] = sum of vals[e] over edges e with group==n.
    vals [E, F] -> out [N, F].

    Implemented as a blocked dense matmul against an on-the-fly one-hot incidence:
    per edge block, ``OH[n, j] = (group[j] == n)`` and ``out += OH @ vals_block``. The
    matmul accumulates in fp32 (so no bf16 prefix-sum cancellation), the one-hot is built
    transiently by a broadcast compare (never a full [N, E] constant), and there is no
    scatter. The one-hot is bf16 (1.0/0.0 are exact) to halve its footprint; XLA keeps
    every unrolled block's one-hot live, so bytes-per-element -- not block size -- is the
    memory lever."""
    E, Fd = vals.shape
    dev = vals.device
    ar = torch.arange(num_nodes, device=dev).unsqueeze(1)  # [N, 1]
    out = torch.zeros(num_nodes, Fd, dtype=torch.float32, device=dev)
    for st in range(0, E, block):
        en = min(st + block, E)
        oh = (ar == group_idx[st:en].unsqueeze(0)).to(torch.bfloat16)  # [N, b]
        out = out + (oh @ vals[st:en].to(torch.bfloat16)).to(torch.float32)
    return out.to(vals.dtype)


class SegmentSum(torch.autograd.Function):
    """edge->node sum. fwd = blocked-matmul segsum; bwd = gather (grad_node[group])."""

    @staticmethod
    def forward(ctx, vals, edge_idx, num_nodes):
        ctx.save_for_backward(edge_idx)
        return _segsum(vals, edge_idx, num_nodes)

    @staticmethod
    def backward(ctx, grad_out):
        (edge_idx,) = ctx.saved_tensors
        return grad_out.index_select(0, edge_idx), None, None


class EdgeGather(torch.autograd.Function):
    """node->edge gather. fwd = index_select; bwd = blocked-matmul segsum (overrides the
    default scatter_add backward of index_select, which would reintroduce scatter)."""

    @staticmethod
    def forward(ctx, node_feat, edge_idx, num_nodes):
        ctx.save_for_backward(edge_idx)
        ctx.num_nodes = num_nodes
        return node_feat.index_select(0, edge_idx)

    @staticmethod
    def backward(ctx, grad_edge):
        (edge_idx,) = ctx.saved_tensors
        return _segsum(grad_edge, edge_idx, ctx.num_nodes), None, None


class ConstMatmul(torch.autograd.Function):
    """y = x @ W for a CONSTANT W, with an explicit reshape-free backward
    (grad = gy @ W.t()) so autograd does not insert a reshape that ttnn rejects on
    tile-padded sub-32 dims."""

    @staticmethod
    def forward(ctx, x, W):
        ctx.save_for_backward(W)
        return x @ W

    @staticmethod
    def backward(ctx, gy):
        (W,) = ctx.saved_tensors
        return gy @ W.t(), None


class SpMMGATv2Conv(GATv2Conv):
    """``GATv2Conv`` whose forward is rewritten with SpMM (matmul) message passing so it
    emits zero scatter and runs on TT. Reuses ``lin_l``/``lin_r``/``bias``; ``att`` is
    stored flat as [1, H*C]. Call :meth:`set_graph` (with self-loops baked into the edge
    index) before forward; the ``edge_index`` argument to forward is ignored."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_self_loops = False  # caller bakes self-loops into the edge index
        with torch.no_grad():
            flat = self.att.detach().reshape(1, self.heads * self.out_channels).clone()
        self.att = nn.Parameter(flat)

    def set_graph(self, src, dst, num_nodes):
        """Provide the (self-looped) edge endpoints and node count, and build the constant
        per-head reduction matrix M [H*C, H] (block-ones summing channels within a head)."""
        self._src, self._dst, self._N = src, dst, num_nodes
        H, C = self.heads, self.out_channels
        M = torch.zeros(H * C, H, device=self.att.device, dtype=self.att.dtype)
        for h in range(H):
            M[h * C : (h + 1) * C, h] = 1.0
        self._M = M
        self._Mt = M.t().contiguous()

    def forward(self, x, edge_index=None):
        H, N = self.heads, self._N
        M = self._M.to(x.dtype)
        x_l = self.lin_l(x)  # [N, H*C]
        x_r = x_l if self.share_weights else self.lin_r(x)  # [N, H*C]
        xj = EdgeGather.apply(x_l, self._src, N)  # [E, H*C]  x_l[src]
        xi = EdgeGather.apply(x_r, self._dst, N)  # [E, H*C]  x_r[dst]
        s = F.leaky_relu(xi + xj, self.negative_slope)  # [E, H*C]
        alpha = ConstMatmul.apply(s * self.att, M)  # [E, H]  per-head sum
        ex = torch.exp(alpha - alpha.detach().max())  # global-max stabilize
        den_node = SegmentSum.apply(ex, self._dst, N)  # [N, H]
        den_edge = EdgeGather.apply(den_node, self._dst, N)  # [E, H]
        alpha = ex / (den_edge + 1e-16)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = xj * ConstMatmul.apply(alpha, self._Mt)  # [E, H*C] broadcast/head
        out = SegmentSum.apply(msg, self._dst, N)  # [N, H*C]  aggregate
        if not self.concat and H > 1:
            out = ConstMatmul.apply(out, M) / H  # head-mean, reshape-free
        if self.bias is not None:
            out = out + self.bias
        return out


def setup_graph(convs, edge_index, num_nodes):
    """Add self-loops once and bind the graph to every SpMMGATv2Conv in ``convs``."""
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    src, dst = edge_index[0].contiguous(), edge_index[1].contiguous()
    for conv in convs:
        conv.set_graph(src, dst, num_nodes)


def masked_nll_loss(out, y, mask):
    """NLL loss over masked nodes using a static float mask (no boolean-advanced
    indexing, whose dynamic-shape backward ttnn rejects). Equivalent to
    ``F.nll_loss(out[mask], y[mask])``."""
    mf = mask.to(out.dtype)
    nll = F.nll_loss(out, y, reduction="none")
    return (nll * mf).sum() / mf.sum()


def masked_accuracy(out, y, mask):
    """Accuracy over masked nodes using a static float mask."""
    mf = mask.to(out.dtype)
    correct = (out.argmax(dim=1) == y).to(out.dtype)
    return (correct * mf).sum() / mf.sum()
