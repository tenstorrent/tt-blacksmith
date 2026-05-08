# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from blacksmith.tools.jax.easydel.workaround_utils import apply_gqa_workaround
from blacksmith.tools.templates.configs import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShardingSpecs:
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

        if self.device_kind == "tt":
            apply_gqa_workaround()

        self.mesh = self._create_mesh()
        self.sharding = self._build_sharding_specs()

    def _setup_env(self) -> None:
        if not self.config.use_tt:
            return

        os.environ.setdefault("PJRT_DEVICE", "TT")
        os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

        if getattr(self.config, "num_devices", 1) > 1:
            os.environ.setdefault("XLA_ALWAYS_ALLREDUCE", "1")
            os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
            os.environ.setdefault("DISABLE_NUMERIC_CC_TOKEN", "1")
            jax.config.update("jax_use_shardy_partitioner", True)

    def _select_device(self) -> tuple[jax.Device, str]:
        """Pick the preferred device: TT > GPU > CPU."""
        try:
            if self.config.use_tt:
                tt_devs = jax.devices("tt")
                if tt_devs:
                    return tt_devs[0], "tt"
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
        """Place a batch on-device, sharding if DP."""
        if self.is_data_parallel():
            return jax.tree.map(
                lambda x: jax.device_put(x, self.sharding.data_sharding),
                batch,
            )
        return jax.tree.map(
            lambda x: jax.device_put(x, self.device),
            batch,
        )

    def replicate(self, pytree):
        return jax.tree.map(
            lambda x: jax.device_put(x, self.sharding.param_sharding),
            pytree,
        )

    @staticmethod
    def to_cpu(pytree):
        cpu = jax.devices("cpu")[0]
        return jax.tree.map(lambda x: jax.device_put(x, cpu), pytree)

    def to_device(self, pytree):
        return jax.tree.map(
            lambda x: jax.device_put(x, self.device),
            pytree,
        )

    def optimizer_step(
        self,
        tx,
        opt_state,
        params,
        grads,
    ):
        """Apply an optax update.

        When optimizer_on_cpu is True and the device is TT, params/grads/opt_state
        are moved to CPU for the update and replicated back afterwards.
        """
        on_cpu = getattr(self.config, "optimizer_on_cpu", True) and self.device_kind == "tt"

        if on_cpu:
            cpu = jax.devices("cpu")[0]
            with jax.default_device(cpu):
                params_c = self.to_cpu(params)
                grads_c = self.to_cpu(grads)
                opt_c = self.to_cpu(opt_state)
                updates, new_opt = tx.update(grads_c, opt_c, params_c)
                new_params = optax.apply_updates(params_c, updates)
            new_params = self.replicate(new_params)
            new_opt = self.replicate(new_opt)
            return new_params, new_opt

        updates, new_opt = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt

    def easydel_load_axis_size(self) -> int:
        """Return the axis size EasyDel should use for sharding_axis_dims at load time.

        On TT this is the total number of TT devices visible to JAX, not the mesh size.
        """
        if self.device_kind == "tt":
            try:
                return len(jax.devices("tt"))
            except Exception:
                return 1
        return 1

    def describe(self) -> dict:
        """Summary dict suitable for TrainingLogger.log_model_info."""
        return {
            "device": self.device_kind,
            "num_devices": getattr(self.config, "num_devices", 1),
            "mesh_shape": list(self.mesh.shape.values()),
            "mesh_axis_names": list(self.mesh.shape.keys()),
            "data_parallel": self.is_data_parallel(),
            "optimizer_on_cpu": (getattr(self.config, "optimizer_on_cpu", True) and self.device_kind == "tt"),
        }

    def __repr__(self) -> str:
        return (
            f"JaxDeviceManager("
            f"device={self.device_kind!r}, "
            f"num_devices={getattr(self.config, 'num_devices', 1)}, "
            f"mesh={dict(self.mesh.shape)})"
        )
