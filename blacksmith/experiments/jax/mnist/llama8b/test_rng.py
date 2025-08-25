#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal test for TT device RNG initialization issue
"""
import sys
from pathlib import Path

# Add the parent directory of blacksmith to Python path
sys.path.insert(0, "/localdev/upantelic/tt-blacksmith")

from blacksmith.tools.jax_utils import init_device
import jax
import jax.random as random

# Initialize TT device
print("🚀 Initializing TT device...")
init_device()

# Get devices
tt_devices = jax.devices("tt")
cpu_devices = jax.devices("cpu")

print(f"TT devices: {tt_devices}")
print(f"CPU devices: {cpu_devices}")

# Test RNG on CPU (should work)
print("\n✅ Testing RNG on CPU...")
with jax.default_device(cpu_devices[0]):
    cpu_key = random.PRNGKey(42)
    print(f"CPU RNG key: {cpu_key}")

# Test RNG on TT device (might fail)
print("\n🎯 Testing RNG on TT device...")
try:
    with jax.default_device(tt_devices[0]):
        tt_key = random.PRNGKey(42)
        print(f"TT RNG key: {tt_key}")
        print("✅ TT RNG works!")
except Exception as e:
    print(f"❌ TT RNG failed: {e}")

print("Done.")
