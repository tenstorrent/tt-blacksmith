#!/usr/bin/env python3
"""
Minimal JAX + Tenstorrent PJRT repro for multi-device (2-chip) jit issues.

Background (seen with tt-blacksmith EasyDeL Qwen on 2 TT devices):
  AttributeError: 'UnspecifiedValue' object has no attribute
  'addressable_devices_indices_map'
  inside jax._src.array._array_shard_arg when entering a jitted train/fwd step.

This script depends only on JAX + NumPy — no EasyDeL imports, no model load.
It builds a 2-device mesh, places arrays with NamedSharding, and runs small
jitted functions. If your stack hits the bug, one of the cases below should
fail during the first jit compile or first execution.
"""

from __future__ import annotations

import sys
import traceback


def _need_two_tt_devices():
    import jax

    try:
        tts = jax.devices("tt")
    except Exception as e:  # pragma: no cover - environment-specific
        print(f"Could not list jax.devices('tt'): {e}")
        return None
    if len(tts) < 2:
        print(
            f"Need at least 2 Tenstorrent devices (jax.devices('tt')); got {len(tts)}. "
            "Set TT_VISIBLE_DEVICES or run on a 2-chip host."
        )
        return None
    return tts


def case_replicated_only(mesh, jnp, NamedSharding, P):
    """Control: replicated bf16 arrays, trivial jit."""
    import jax

    rep = NamedSharding(mesh, P())

    @jax.jit
    def f(a, b):
        return jnp.sum(a * b)

    a = jax.device_put(jnp.ones((8, 8), dtype=jnp.bfloat16), rep)
    b = jax.device_put(jnp.ones((8, 8), dtype=jnp.bfloat16), rep)
    with mesh:
        out = f(a, b)
        out.block_until_ready()
    print(f"  case_replicated_only: ok, out={float(out)}")


def case_sharded_elementwise(mesh, jnp, NamedSharding, P):
    """Shard a 1-D vector across the mesh axis; elementwise add inside jit."""
    import jax

    shard = NamedSharding(mesh, P("x"))

    @jax.jit
    def f(u, v):
        return jnp.sum((u + v).astype(jnp.float32))

    u = jax.device_put(jnp.arange(8.0, dtype=jnp.bfloat16), shard)
    v = jax.device_put(jnp.ones(8, dtype=jnp.bfloat16), shard)
    with mesh:
        out = f(u, v)
        out.block_until_ready()
    print(f"  case_sharded_elementwise: ok, out={float(out)}")


def case_mixed_pytree_matmul(mesh, jnp, NamedSharding, P):
    """
    Closer to training: dict state + dict batch, weight rows sharded,
    batch replicated — forces shard-arg handling across a pytree boundary.
    """
    import jax

    s_w = NamedSharding(mesh, P("x", None))
    s_rep = NamedSharding(mesh, P())

    @jax.jit
    def step(params, batch):
        w = params["w"]
        x = batch["x"]
        return jnp.sum(jax.nn.tanh(x @ w).astype(jnp.float32))

    params = {
        "w": jax.device_put(jnp.ones((32, 32), dtype=jnp.bfloat16), s_w),
    }
    batch = {
        "x": jax.device_put(jnp.ones((4, 32), dtype=jnp.bfloat16), s_rep),
    }
    with mesh:
        out = step(params, batch)
        out.block_until_ready()
    print(f"  case_mixed_pytree_matmul: ok, out={float(out)}")


def main() -> int:
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    tts = _need_two_tt_devices()
    if tts is None:
        return 1

    mesh = jax.make_mesh((2,), ("x",), devices=(tts[0], tts[1]))
    print(f"Mesh devices: {mesh.devices.flatten().tolist()}")

    cases = (
        ("replicated_only", case_replicated_only),
        ("sharded_elementwise", case_sharded_elementwise),
        ("mixed_pytree_matmul", case_mixed_pytree_matmul),
    )
    for name, fn in cases:
        print(f"Running {name} ...")
        try:
            fn(mesh, jnp, NamedSharding, P)
        except Exception:
            print(f"  FAILED: {name}")
            traceback.print_exc()
            return 2

    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
