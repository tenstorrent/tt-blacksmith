# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np

import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding


class ShardingConfig:
    def __init__(self):
        self.mesh = Mesh(np.array(jax.devices("tt")), axis_names=("data",))
        self.data_partition = PartitionSpec("data")
        self.param_partition = PartitionSpec()
        self.data_sharding = NamedSharding(self.mesh, self.data_partition)
        self.param_sharding = NamedSharding(self.mesh, self.param_partition)
