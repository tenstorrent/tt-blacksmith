# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TT PJRT ignores NumPy strides: jnp.asarray scrambles non-contiguous arrays.

Usage:  python repro_stride_bug.py          # TT (reproduces bug)
        python repro_stride_bug.py --cpu    # CPU (both pass)
"""

import os, sys  # noqa: E401

if "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ.setdefault("PJRT_DEVICE", "TT")
    os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

import jax.numpy as jnp  # noqa: E402
import numpy as np        # noqa: E402

data = np.arange(12, dtype=np.float32).reshape(3, 4)
fortran_order = data.T  # non-contiguous view (like PyTorch's permute(1, 0))
assert not fortran_order.flags.c_contiguous

expected = np.ascontiguousarray(fortran_order)
buggy = np.array(jnp.asarray(fortran_order))
fixed = np.array(jnp.asarray(np.ascontiguousarray(fortran_order)))

print(f"Expected:\n{expected}\n\nBuggy:\n{buggy}\n\nFixed:\n{fixed}\n")
print(f"Buggy matches: {np.array_equal(buggy, expected)}")
print(f"Fixed matches: {np.array_equal(fixed, expected)}")
