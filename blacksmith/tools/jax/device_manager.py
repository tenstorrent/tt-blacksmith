# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from blacksmith.tools.jax.easydel.partitioning import build_param_partition_specs
from blacksmith.tools.jax.easydel.workaround_utils import apply_gqa_workaround, apply_lora_workaround
from blacksmith.tools.templates.configs import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShardingSpecs:
    """Immutable container for mesh and sharding objects."""

    mesh: Mesh
    data_partition: PartitionSpec
    param_partition: PartitionSpec
    data_sharding: NamedSharding
    param_sharding: NamedSharding


class JaxDeviceManager:
    """Manage JAX devices, environment, mesh, and sharding for JAX/EasyDel training."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

        self._setup_env()

        self.device, self.device_kind = self._select_device()
        jax.config.update("jax_default_device", self.device)

        if getattr(config, "apply_gqa_workaround", True) and self.device_kind == "tt":
            apply_gqa_workaround()

        if getattr(config, "apply_lora_workaround", True) and self.device_kind == "tt":
            apply_lora_workaround()

        self.mesh = self._create_mesh()
        self.sharding = self._build_sharding_specs()

    def _setup_env(self) -> None:
        """Set TT-XLA environment variables."""
        if not self.config.use_tt:
            return

        os.environ.setdefault("PJRT_DEVICE", "TT")
        os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

        if getattr(self.config, "num_devices", 1) > 1:
            os.environ.setdefault("XLA_ALWAYS_ALLREDUCE", "1")
            os.environ.setdefault("DISABLE_NUMERIC_CC_TOKEN", "1")
            use_shardy = getattr(self.config, "use_shardy_partitioner", True)
            os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1" if use_shardy else "0")
            jax.config.update("jax_use_shardy_partitioner", use_shardy)

    def _select_device(self) -> tuple[jax.Device, str]:
        """Pick the preferred device: TT > GPU > CPU."""
        if self.config.use_tt:
            try:
                tt_devs = jax.devices("tt")
                if tt_devs:
                    return tt_devs[0], "tt"
            except Exception:
                pass

        try:
            gpu_devs = jax.devices("gpu")
            if gpu_devs:
                return gpu_devs[0], "gpu"
        except Exception:
            pass

        return jax.devices("cpu")[0], "cpu"

    def _create_mesh(self) -> Mesh:
        """Build a JAX mesh, defaulting to a single-axis ('data',) layout."""
        n = getattr(self.config, "num_devices", 1)
        shape = tuple(getattr(self.config, "mesh_shape", None) or [n])
        names = tuple(getattr(self.config, "mesh_axis_names", None) or ["data"])

        devices = tuple(jax.devices(self.device_kind)[:n])

        return jax.make_mesh(shape, names, devices=devices)

    def _build_sharding_specs(self) -> ShardingSpecs:
        """Derive data / param sharding objects from config."""
        dim = getattr(self.config, "input_sharding_dim", "data")
        if dim is not None and getattr(self.config, "num_devices", 1) > 1:
            data_ps = PartitionSpec(dim)
        else:
            data_ps = PartitionSpec()

        param_ps = PartitionSpec()

        return ShardingSpecs(
            mesh=self.mesh,
            data_partition=data_ps,
            param_partition=param_ps,
            data_sharding=NamedSharding(self.mesh, data_ps),
            param_sharding=NamedSharding(self.mesh, param_ps),
        )

    def is_data_parallel(self) -> bool:
        """True when data is sharded across >1 device."""
        return (
            getattr(self.config, "input_sharding_dim", "data") is not None
            and getattr(self.config, "num_devices", 1) > 1
        )

    def prepare_batch(
        self,
        batch: dict[str, jnp.ndarray],
    ) -> dict[str, jax.Array]:
        """Place a batch on-device.

        DP: shard along input_sharding_dim via explicit host-side slicing.
        TP / replicated: replicate across every chip.
        Single device: place on the selected device.
        """
        if self.is_data_parallel():
            sharding = self.sharding.data_sharding
            out = jax.tree.map(lambda x: self._place_sharded(x, sharding), batch)
        elif getattr(self.config, "num_devices", 1) > 1:
            target = NamedSharding(self.mesh, PartitionSpec())
            out = jax.tree.map(lambda x: jax.device_put(x, target), batch)
        else:
            out = jax.tree.map(lambda x: jax.device_put(x, self.device), batch)

        jax.block_until_ready(out)
        return out

    def _place_sharded(self, leaf, sharding: NamedSharding) -> jax.Array:
        """Place a single array on-mesh using explicit per-device HtoD transfers."""
        host_np = np.asarray(leaf)
        indices_map = sharding.devices_indices_map(host_np.shape)
        shard_arrays = []
        for device in sharding.mesh.devices.flatten():
            shard_np = np.ascontiguousarray(host_np[indices_map[device]])
            shard_arrays.append(jax.device_put(shard_np, device))
        return jax.make_array_from_single_device_arrays(host_np.shape, sharding, shard_arrays)

    def replicate(self, pytree):
        """Replicate a pytree across the mesh (no sharding)."""
        out = jax.tree.map(
            lambda x: jax.device_put(x, self.sharding.param_sharding),
            pytree,
        )
        jax.block_until_ready(out)
        return out

    def apply_sharding_patterns(self, pytree, patterns):
        """Place leaves of pytree using yaml regex->PartitionSpec patterns.

        Each entry in patterns is [regex, [axis_or_null, ...]]. The first
        matching regex wins; unmatched leaves are replicated. Falls back
        to replicate() when patterns is empty.
        """
        if not patterns:
            return self.replicate(pytree)

        rules = [(entry[0], PartitionSpec(*entry[1])) for entry in patterns]
        specs = build_param_partition_specs(pytree, rules, default=PartitionSpec())
        leaves = jax.tree_util.tree_leaves(specs)
        matched = sum(1 for s in leaves if s != PartitionSpec())
        logger.info(f"apply_sharding_patterns: {matched}/{len(leaves)} leaves matched a TP rule")

        def _place(leaf, spec):
            sharding = NamedSharding(self.mesh, spec)
            if spec == PartitionSpec():
                return jax.device_put(leaf, sharding)
            host_np = np.asarray(leaf)
            indices_map = sharding.devices_indices_map(leaf.shape)
            shard_arrays = []
            for device in sharding.mesh.devices.flatten():
                shard_np = np.ascontiguousarray(host_np[indices_map[device]])
                shard_arrays.append(jax.device_put(shard_np, device))
            return jax.make_array_from_single_device_arrays(leaf.shape, sharding, shard_arrays)

        placed = jax.tree.map(_place, pytree, specs)
        jax.block_until_ready(placed)
        return placed

    @staticmethod
    def to_cpu(pytree):
        """Move every leaf of pytree to CPU."""
        cpu = jax.devices("cpu")[0]
        return jax.tree.map(lambda x: jax.device_put(x, cpu), pytree)

    def to_device(self, pytree):
        """Move every leaf onto self.device."""
        return jax.tree.map(
            lambda x: jax.device_put(x, self.device),
            pytree,
        )

    def easydel_load_axis_size(self) -> int:
        """Return the axis size EasyDel should use for sharding_axis_dims at load time."""
        return getattr(self.config, "num_devices", 1)

    def describe(self) -> dict:
        """Summary dict suitable for TrainingLogger.log_model_info."""
        return {
            "device": self.device_kind,
            "num_devices": getattr(self.config, "num_devices", 1),
            "mesh_shape": list(self.mesh.shape.values()),
            "mesh_axis_names": list(self.mesh.shape.keys()),
            "data_parallel": self.is_data_parallel(),
            "gqa_workaround": (getattr(self.config, "apply_gqa_workaround", True) and self.device_kind == "tt"),
        }

    def __repr__(self) -> str:
        return (
            f"JaxDeviceManager("
            f"device={self.device_kind!r}, "
            f"num_devices={getattr(self.config, 'num_devices', 1)}, "
            f"mesh={dict(self.mesh.shape)})"
        )
