# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal reproducer: ttnn.matmul float32 vs bfloat16 accumulator precision.

Tests whether ttnn matmul uses float32 accumulators for bfloat16 inputs
by comparing error at different reduction dimensions. If accumulators are
bfloat16, error scales ~linearly with K. If float32, error scales ~sqrt(K).

Also tests float32 matmul directly to check if float32 is generally broken.

Usage:
    python test_ttnn_matmul.py
"""

import torch
import ttnn

device = ttnn.open_device(device_id=0)

torch.manual_seed(42)


def test_matmul(name, K, N, dtype):
    """Run matmul (1, 5, K) @ (K, N) on TT and compare against torch CPU."""
    x = torch.randn(1, 5, K)
    w = torch.randn(K, N)

    if dtype == torch.bfloat16:
        x_ref = x.bfloat16().float()
        w_ref = w.bfloat16().float()
    else:
        x_ref = x.clone()
        w_ref = w.clone()
    expected = x_ref @ w_ref

    x_tt = ttnn.from_torch(x.to(dtype), device=device, layout=ttnn.TILE_LAYOUT)
    w_tt = ttnn.from_torch(w.to(dtype), device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.to_torch(x_tt @ w_tt).float()

    diff = (result - expected).abs().max().item()
    mean = (result - expected).abs().mean().item()
    status = "PASS" if diff < 1.0 else "FAIL"
    print(f"  [{status}] {name:<35} K={K:<6} max_diff={diff:<10.4f} mean={mean:.6f}")
    return diff


print("=== bfloat16 matmul (varying reduction dim) ===")
for K in [64, 256, 896, 2048, 4096, 8192]:
    test_matmul(f"bf16 K={K}", K, 256, torch.bfloat16)

print("\n=== float32 matmul (is float32 broken?) ===")
for K in [64, 256, 896, 2048]:
    test_matmul(f"f32 K={K}", K, 256, torch.float32)

ttnn.close_device(device)
