# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""JAX mesh and :class:`~jax.sharding.NamedSharding` for TT multi-chip EasyDeL Qwen.

Same layout as ``jax/distil_bert/multi_chip/data_parallel/sharding_config.py``:
``PartitionSpec()`` for replicated tensors on the mesh.  Here both *param* and
*data* use replication (workaround for PJRT ``UnspecifiedValue`` on multidevice
``jit``; see ``train_steps``).  Axis name **X** matches ``load_model`` and
``make_tt_mesh``.
"""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# Must match ``sharding_axis_names`` in :func:`test_qwen_fine_tuning_easydel.load_model`.
AXIS_NAME = "X"


def make_tt_mesh(num_devices: int, device_kind: str = "tt") -> Mesh:
    """Build a 1D mesh over the first *num_devices* accelerators of *device_kind*."""
    devices = tuple(jax.devices(device_kind)[:num_devices])
    return jax.make_mesh((num_devices,), (AXIS_NAME,), devices=devices)


class ShardingConfig:
    """Replicated parameters and batch tensors on *mesh* (full ``PartitionSpec()``)."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.param_partition = PartitionSpec()
        self.data_partition = PartitionSpec()
        self.param_sharding = NamedSharding(self.mesh, self.param_partition)
        self.data_sharding = NamedSharding(self.mesh, self.data_partition)
