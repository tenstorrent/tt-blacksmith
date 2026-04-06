# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal reproducer: ttnn.reshape 4D -> 5D gives wrong results after to_layout.

When a float32 tensor in ROW_MAJOR layout is converted to TILE layout
and then reshaped from 4D to 5D, the output is incorrect (max_diff=7.0).

Usage:
    python test_ttnn_reshape.py

Expected output:
    [PASS] max_diff=0.0

Actual output:
    [FAIL] max_diff=7.0
"""

import torch
import ttnn

device = ttnn.open_device(device_id=0)

x = torch.arange(1 * 5 * 16 * 128, dtype=torch.float32).reshape(1, 5, 16, 128)
expected = x.reshape(1, 5, 8, 2, 128)

x_tt = ttnn.from_torch(x, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
x_tt = ttnn.to_layout(x_tt, layout=ttnn.TILE_LAYOUT)
result = ttnn.to_torch(ttnn.reshape(x_tt, (1, 5, 8, 2, 128)))

diff = (result.float() - expected.float()).abs().max().item()
print(f"[{'PASS' if diff == 0.0 else 'FAIL'}] max_diff={diff}")

ttnn.close_device(device)
