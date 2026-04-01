# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test bfloat16 matmul precision vs reduction dimension size.

If TT uses bfloat16 accumulators, error will scale with reduction dim.
If TT uses float32 accumulators, error will stay roughly constant.

Usage:
    JAX_PLATFORMS=cpu python test_matmul.py   # CPU baseline
    python test_matmul.py                      # TT device
"""

import os

os.environ.setdefault("PJRT_DEVICE", "TT")
os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def matmul(x, w):
    return jnp.dot(x, w)


def test_reduction_dim(rng, K, N=256):
    """Test matmul (5, K) @ (K, N) and return max_diff, mean_diff."""
    x_np = rng.standard_normal((5, K)).astype(np.float32)
    w_np = rng.standard_normal((K, N)).astype(np.float32)

    x = jnp.array(x_np, dtype=jnp.bfloat16)
    w = jnp.array(w_np, dtype=jnp.bfloat16)

    result = np.array(matmul(x, w), dtype=np.float32)

    x_bf = np.array(x, dtype=np.float32)
    w_bf = np.array(w, dtype=np.float32)
    expected = x_bf @ w_bf

    max_diff = np.max(np.abs(result - expected))
    mean_diff = np.mean(np.abs(result - expected))
    return max_diff, mean_diff


def main():
    platform = jax.devices()[0].platform
    print(f"Platform: {platform}\n")

    rng = np.random.default_rng(42)
    dims = [64, 128, 256, 512, 896, 1024, 2048, 4096, 8192]

    print(f"{'K (reduction dim)':<20} {'max_diff':>10} {'mean_diff':>12}")
    print("-" * 44)

    for K in dims:
        max_d, mean_d = test_reduction_dim(rng, K)
        print(f"{K:<20} {max_d:>10.4f} {mean_d:>12.6f}")

    print("\nIf max_diff scales ~linearly with K → bfloat16 accumulators (bad)")
    print("If max_diff stays roughly constant  → float32 accumulators (good)")


if __name__ == "__main__":
    main()
