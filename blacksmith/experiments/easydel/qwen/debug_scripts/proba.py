# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal TT multichip **data-parallel** JAX template (not production training).

Mirrors ``jax/mnist/multi_chip/data_parallel/test_pure_jax_mnist.py``:
``Mesh`` → ``NamedSharding`` / ``PartitionSpec`` → CPU init → ``device_put`` →
``shard_map`` + ``jit`` → host batch loop. Replace ``toy_forward`` with your model.

**Not covered here:** tensor parallel (see ``jax/mnist/multi_chip/tensor_parallel/``),
EasyDeL mesh rules, or full LoRA/optax wiring.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util
from jax.experimental import shard_map
from jax.lax import all_gather
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Mesh axis name (must match ``all_gather(..., AXIS)`` below).
AXIS = "dp"


def make_mesh_tt():
    devs = jax.devices("tt")
    return Mesh(np.array(devs).reshape(len(devs),), (AXIS,))


class DPSharding:
    """DP: replicated params (``P()``), sharded batch on ``AXIS``."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.param = NamedSharding(mesh, P())
        self.data = NamedSharding(mesh, P(AXIS))


def toy_params(key, din, dout):
    with jax.default_device(jax.devices("cpu")[0]):
        k0, k1 = jax.random.split(key)
        w = jax.random.normal(k0, (din, dout))
        b = jax.random.normal(k1, (dout,))
        return w, b


def toy_forward(params, x):
    w, b = params
    return x @ w + b


def _local_loss_grads(params, x, y):
    def loss_fn(p):
        return jnp.mean((toy_forward(p, x) - y) ** 2)

    loss, grads = jax.value_and_grad(loss_fn)(params)

    def mean_grad(g):
        return jnp.mean(all_gather(g, AXIS), axis=0)

    grads = tree_util.tree_map(mean_grad, grads)
    return loss, grads


def train_step_shard_map(mesh: Mesh):
    return shard_map.shard_map(
        lambda p, x, y, lr: _sgd_step(p, x, y, lr),
        mesh=mesh,
        in_specs=(P(), P(AXIS), P(AXIS), P()),
        out_specs=(P(), P()),
        check_rep=False,
    )


def _sgd_step(params, x, y, lr):
    _, grads = _local_loss_grads(params, x, y)
    return tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def main():
    jax.config.update("jax_use_shardy_partitioner", True)
    mesh = make_mesh_tt()
    sh = DPSharding(mesh)
    key = jax.random.key(0)

    with jax.default_device(jax.devices("cpu")[0]):
        params_cpu = toy_params(key, din=8, dout=4)
    params = jax.device_put(params_cpu, sh.param)

    # Host batch → shard on leading dim across ``AXIS`` (each chip sees batch/N rows)
    x_host = np.random.randn(32, 8).astype(np.float32)
    y_host = np.random.randn(32, 4).astype(np.float32)
    x = jax.device_put(x_host, sh.data)
    y = jax.device_put(y_host, sh.data)
    lr = jax.device_put(np.float32(0.01), sh.param)

    step = train_step_shard_map(mesh)
    step_jit = jax.jit(step)

    new_params = step_jit(params, x, y, lr)
    print(jax.device_get(new_params)[0].shape)


if __name__ == "__main__":
    main()
