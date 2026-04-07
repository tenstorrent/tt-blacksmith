# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test: does a simple 4D -> 5D reshape produce correct results on TT?

Compares JAX's reshape output against NumPy's reshape (ground truth).

Run on CPU:   JAX_PLATFORMS=cpu python test_5d_reshape.py
Run on TT:    python test_5d_reshape.py
"""

import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def reshape_4d_to_5d(x):
    b, s, h, d = x.shape
    return x.reshape(b, s, h // 2, 2, d)


def main():
    platform = jax.devices()[0].platform
    print(f"DEVICE: {jax.devices()[0]}")
    print(f"Platform: {platform}")

    x_np = np.arange(1 * 5 * 16 * 128, dtype=np.float32).reshape(1, 5, 16, 128)
    x_jax = jnp.array(x_np)

    result = np.array(reshape_4d_to_5d(x_jax), dtype=np.float32)
    expected = x_np.reshape(1, 5, 8, 2, 128)

    diff = np.max(np.abs(result - expected))
    print(f"max_diff: {diff:.6e}")
    print("PASS" if diff == 0.0 else "FAIL")


if __name__ == "__main__":
    main()
