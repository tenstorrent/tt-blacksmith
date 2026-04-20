# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Concrete JAX data-parallel sharding config for TT multi-chip EasyDeL Qwen.

Same shape as DistilBERT data-parallel config:
- parameters are replicated (`PartitionSpec()`)
- input/label tensors are sharded on axis `data` (`PartitionSpec("data")`)
"""

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

# Matches DistilBERT data-parallel axis.
AXIS_NAME = "data"


def make_tt_mesh(num_devices: int, device_kind: str = "tt") -> Mesh:
    """Build a 1D mesh over the first *num_devices* accelerators of *device_kind*.

    The axis is always :data:`AXIS_NAME` (``"data"``) so single-chip and
    multi-chip paths share the same mesh axis name.
    """
    devices = tuple(jax.devices(device_kind)[:num_devices])
    return jax.make_mesh((num_devices,), (AXIS_NAME,), devices=devices)


class ShardingConfig:
    """Container for mesh, PartitionSpec, and NamedSharding objects."""

    def __init__(self, num_devices: int, device_kind: str = "tt"):
        devices = tuple(jax.devices(device_kind)[:num_devices])
        self.mesh: Mesh = Mesh(np.array(devices), axis_names=(AXIS_NAME,))
        self.param_partition = PartitionSpec()
        self.data_partition = PartitionSpec(AXIS_NAME)
        self.param_sharding = NamedSharding(self.mesh, self.param_partition)
        self.data_sharding = NamedSharding(self.mesh, self.data_partition)
