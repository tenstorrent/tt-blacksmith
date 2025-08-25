#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simple TT Device Initialization

Minimal script to initialize TT device using jax_utils.init_device()
"""

from jax_utils import init_device

init_device()
import jax
import jax.numpy as jnp


def main():
    print("🚀 Simple TT Device Initialization")
    print("-" * 40)

    print("Before init:")
    print(f"  Devices: {jax.devices()}")
    print(f"  Default backend: {jax.default_backend()}")

    try:
        print("\n⚡ Initializing TT device...")

        print("\nAfter init:")
        print(f"  Devices: {jax.devices()}")
        print(f"  Default backend: {jax.default_backend()}")

        # Quick test
        print(f"\n🧮 Quick test:")
        x = jnp.array([1.0, 2.0, 3.0])
        result = x * 2
        print(f"  x * 2 = {result}")
        print(f"  Device: {result.device()}")

        print("\n✅ TT device ready!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Falling back to CPU...")


if __name__ == "__main__":
    main()
