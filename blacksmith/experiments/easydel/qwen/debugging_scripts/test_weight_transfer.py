# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test whether pretrained weights get corrupted during transfer to TT device.

Loads Qwen3-0.6B q_proj weight on CPU, then tests two transfer paths:
  1. Direct: jnp.array(weight) on TT → read back → compare
  2. JIT:   jax.jit(identity)(weight) on TT → read back → compare

This isolates whether corruption comes from layout conversion or JIT pipeline.
"""

import os

os.environ.setdefault("PJRT_DEVICE", "TT")
os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoModelForCausalLM


def main():
    platform = jax.devices()[0].platform
    print(f"Platform: {platform}")

    # Load pretrained weight via PyTorch (avoids any JAX/TT involvement)
    print("Loading Qwen3-0.6B via PyTorch...")
    pt_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    q_weight_np = pt_model.model.layers[0].self_attn.q_proj.weight.detach().numpy()
    del pt_model
    print(f"q_proj weight shape: {q_weight_np.shape}, dtype: {q_weight_np.dtype}")
    print(f"CPU range: [{q_weight_np.min():.6f}, {q_weight_np.max():.6f}]")

    # Convert to bfloat16 numpy reference (same as EasyDel would)
    ref_bf16 = np.array(jnp.array(q_weight_np, dtype=jnp.bfloat16), dtype=np.float32)

    # --- Test 1: Direct jnp.array placement ---
    x_tt = jnp.array(q_weight_np, dtype=jnp.bfloat16)
    result_direct = np.array(x_tt, dtype=np.float32)
    diff_direct = np.max(np.abs(result_direct - ref_bf16))
    mean_direct = np.mean(np.abs(result_direct - ref_bf16))
    print(f"\n[Direct transfer]  max_diff={diff_direct:.6f}  mean_diff={mean_direct:.8f}")

    # --- Test 2: Through jax.jit identity ---
    @jax.jit
    def identity(x):
        return x

    x_jit = identity(jnp.array(q_weight_np, dtype=jnp.bfloat16))
    result_jit = np.array(x_jit, dtype=np.float32)
    diff_jit = np.max(np.abs(result_jit - ref_bf16))
    mean_jit = np.mean(np.abs(result_jit - ref_bf16))
    print(f"[JIT identity]     max_diff={diff_jit:.6f}  mean_diff={mean_jit:.8f}")

    # --- Test 3: Through jax.jit matmul (weight as closed-over constant) ---
    w_const = jnp.array(q_weight_np.T, dtype=jnp.bfloat16)  # transposed for x @ w
    dummy_input = jnp.ones((1, 5, q_weight_np.shape[1]), dtype=jnp.bfloat16)

    @jax.jit
    def matmul_with_captured_weight(x):
        return jnp.dot(x, w_const)

    @jax.jit
    def matmul_with_arg_weight(x, w):
        return jnp.dot(x, w)

    cpu = jax.devices("cpu")[0]
    dummy_np = np.array(jax.device_put(dummy_input, cpu), dtype=np.float32)
    w_np = np.array(jax.device_put(w_const, cpu), dtype=np.float32)
    expected = dummy_np @ w_np

    result_captured = np.array(matmul_with_captured_weight(dummy_input), dtype=np.float32)
    diff_captured = np.max(np.abs(result_captured - expected))
    mean_captured = np.mean(np.abs(result_captured - expected))
    print(f"[JIT captured W]   max_diff={diff_captured:.6f}  mean_diff={mean_captured:.8f}")

    result_arg = np.array(matmul_with_arg_weight(dummy_input, w_const), dtype=np.float32)
    diff_arg = np.max(np.abs(result_arg - expected))
    mean_arg = np.mean(np.abs(result_arg - expected))
    print(f"[JIT arg W]        max_diff={diff_arg:.6f}  mean_diff={mean_arg:.8f}")

    print()
    for name, d in [("Direct", diff_direct), ("JIT identity", diff_jit),
                     ("JIT captured W", diff_captured), ("JIT arg W", diff_arg)]:
        print(f"  {name:<18} {'PASS' if d < 1.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
