# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TT-MLIR eliminates bf16→f32 casts inside fused JIT graphs, producing wrong
softmax/cross-entropy over large reduction dimensions (e.g. vocab=151936).

The bug: when a bf16 tensor is PRODUCED inside a @jax.jit (e.g. by a matmul),
then cast to f32 with .astype(jnp.float32), TT-MLIR silently drops the cast.
The subsequent log-softmax reduction over ~150k elements runs in bf16,
accumulating catastrophic rounding errors (~1.5 higher loss).

This does NOT happen when the bf16 tensor is passed as an INPUT to the JIT —
only when it is an intermediate result of computation inside the JIT.

Usage:  python repro_f32_cast_bug.py          # TT (reproduces bug)
        python repro_f32_cast_bug.py --cpu    # CPU (all match)
"""

import os, sys  # noqa: E401

if "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ.setdefault("PJRT_DEVICE", "TT")
    os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

VOCAB = 151936  # Qwen vocabulary size
HIDDEN = 1024   # hidden dimension
SEQ = 127       # shifted sequence length

rng = np.random.RandomState(42)

# Simulate a model's final linear projection (lm_head): hidden → vocab
weight_np = rng.normal(scale=0.02, size=(HIDDEN, VOCAB)).astype(np.float32)
hidden_np = rng.normal(scale=1.0, size=(1, SEQ, HIDDEN)).astype(np.float32)

weight_bf16 = jnp.array(weight_np, dtype=jnp.bfloat16)
hidden_bf16 = jnp.array(hidden_np, dtype=jnp.bfloat16)

# Ground truth: matmul + log_softmax in float64
logits_ref = hidden_np.astype(np.float64) @ weight_np.astype(np.float64)
shifted = logits_ref - logits_ref.max(axis=-1, keepdims=True)
ref_loss = float(-np.mean(shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))))


@jax.jit
def fused_loss_with_cast(hidden, weight):
    """bf16 matmul → .astype(f32) → log_softmax.  Cast SHOULD be preserved."""
    logits = hidden @ weight                      # bf16 matmul → bf16 result
    logits_f32 = logits.astype(jnp.float32)       # explicit cast to f32
    return -jnp.mean(jax.nn.log_softmax(logits_f32, axis=-1))


@jax.jit
def fused_loss_no_cast(hidden, weight):
    """bf16 matmul → log_softmax directly in bf16 (for comparison)."""
    logits = hidden @ weight
    return -jnp.mean(jax.nn.log_softmax(logits, axis=-1))


result_cast = float(fused_loss_with_cast(hidden_bf16, weight_bf16))
result_bf16 = float(fused_loss_no_cast(hidden_bf16, weight_bf16))

print(f"Reference (numpy f64):         {ref_loss:.4f}")
print(f"Fused JIT with .astype(f32):   {result_cast:.4f}  (error: {result_cast - ref_loss:+.4f})")
print(f"Fused JIT native bf16:         {result_bf16:.4f}  (error: {result_bf16 - ref_loss:+.4f})")
print()

cast_err = abs(result_cast - ref_loss)
bf16_err = abs(result_bf16 - ref_loss)

if cast_err < bf16_err * 0.5:
    print("OK: .astype(f32) is more accurate than bf16 — cast was preserved.")
elif cast_err > bf16_err:
    print("BUG: .astype(f32) is LESS accurate than native bf16!")
    print("     f32 should always be at least as precise as bf16.")
    print("     In full transformer models this error grows to ~1.5 on cross-entropy loss.")
else:
    print("BUG: .astype(f32) did not improve accuracy as expected.")
    print(f"     f32 error: {cast_err:.4f},  bf16 error: {bf16_err:.4f}")
