# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Minimal repro for the Wan RoPE *scatter blowup*.

What it shows
-------------
Wan's `apply_rotary_emb` writes the rotated values back into an interleaved
layout with a *strided in-place assignment*:

    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin     # even lanes
    out[..., 1::2] = x1 * sin + x2 * cos     # odd  lanes

torch-xla lowers each strided write to a `scatter` (index_put). tt-mlir then
tiles every scatter into ~256-element chunks, so the op count explodes with the
number of scattered indices ( = B * S * H * D/2 ). At video sequence length one
attention layer alone emits ~80k `ttnn.scatter` ops (+ matching slice/to_layout)
-> ~400k-line TTNN IR.

This script isolates *only* that op (no transformer, no attention, no weights),
so the graph stays tiny and the dumped IR is readable, while still triggering
the exact scatter -> 256-chunk tiling.

Run
---
    # default: tiny shape, scatter (buggy) variant, dumps IR to ./repro_ir/irs
    python repro.py

    # functional (stack+flatten) variant -> no scatter, for op-count comparison
    python repro.py --mode functional

    # crank the sequence length to watch the op count blow up
    REPRO_SEQ=6630 python repro.py --mode scatter

Then compare:
    ls repro_ir/irs/
    grep -c 'ttnn.scatter' repro_ir/irs/ttnn_*_scatter_*.mlir
    grep -c 'ttnn.scatter' repro_ir/irs/ttnn_*_functional_*.mlir   # -> 0
"""

import argparse
import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr

# Tiny by default so the graph + IR stay small. Override via env to scale up and
# watch the scatter count explode. Wan TI2V-5B uses head_dim=128, 12 heads;
# video seq ~6630. The defaults here give B*S*H*D/2 = 1*64*2*32 = 4096 indices,
# enough to produce a handful of 256-chunk scatters (readable), not a monster.
B = int(os.environ.get("REPRO_BATCH", "1"))
S = int(os.environ.get("REPRO_SEQ", "64"))
H = int(os.environ.get("REPRO_HEADS", "2"))
D = int(os.environ.get("REPRO_HEAD_DIM", "64"))  # must be even


class RopeScatter(nn.Module):
    """The buggy upstream interleave: two strided index_put writes -> scatter."""

    def forward(self, x, cos, sin):
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class RopeFunctional(nn.Module):
    """Bit-identical interleave via stack+flatten -> concat+reshape, no scatter."""

    def forward(self, x, cos, sin):
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        even = x1 * cos - x2 * sin
        odd = x1 * sin + x2 * cos
        return torch.stack((even, odd), dim=-1).flatten(-2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["scatter", "functional"],
        default="scatter",
        help="scatter = buggy strided index_put; functional = stack+flatten fix",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="also run a backward pass (dumps g1) to show the scatter/gather VJP",
    )
    parser.add_argument(
        "--export",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="set torch_xla custom compile options to dump vhlo/shlo/ttir/ttnn "
        "MLIR. Use --no-export to skip all custom options (e.g. when relying on "
        "TTXLA_LOGGER_LEVEL=VERBOSE logging instead).",
    )
    args = parser.parse_args()

    assert D % 2 == 0, "head_dim (REPRO_HEAD_DIM) must be even"

    xr.set_device_type("TT")
    device = torch_xla.device()

    # Dump vhlo/shlo/ttir/ttnn MLIR next to the script. The op-count blowup is
    # visible at the ttir->ttnn boundary (one ttir.scatter -> many ttnn.scatter).
    # Skipped under --no-export so the run uses no custom compile options at all
    # (e.g. when capturing TTXLA_LOGGER_LEVEL=VERBOSE output to a file instead).
    if args.export:
        torch_xla.set_custom_compile_options(
            {
                "export_path": "repro_ir",
                "export_model_name": f"rope_{args.mode}_b{B}_s{S}_h{H}_d{D}",
            }
        )

    model = (RopeScatter() if args.mode == "scatter" else RopeFunctional())
    model = model.to(torch.bfloat16).to(device)

    x = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device,
                    requires_grad=args.backward)
    # cos/sin are the de-interleaved halves -> last dim is D/2.
    cos = torch.randn(B, S, H, D // 2, dtype=torch.bfloat16, device=device)
    sin = torch.randn(B, S, H, D // 2, dtype=torch.bfloat16, device=device)

    out = model(x, cos, sin)
    torch_xla.sync()
    if args.backward:
        out.sum().backward()
        torch_xla.sync()

    n_indices = B * S * H * (D // 2)
    print(
        f"mode={args.mode}  shape=({B},{S},{H},{D})  "
        f"strided-write indices per scatter = B*S*H*D/2 = {n_indices}"
    )
    if args.export:
        print("IR dumped to: repro_ir/irs/  (stages: vhlo, shlo, ttir, ttnn)")
        print("Inspect:  grep -c 'ttnn.scatter' repro_ir/irs/ttnn_*.mlir")


if __name__ == "__main__":
    main()
