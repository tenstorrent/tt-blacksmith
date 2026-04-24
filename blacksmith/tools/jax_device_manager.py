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

from blacksmith.tools.templates.configs import TrainingConfig
from blacksmith.tools.workaround_utils_jax import apply_gqa_workaround

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShardingSpecs:
    """Immutable container for mesh + sharding objects.

    Produced by :class:`JaxDeviceManager` at init time so that
    callers never have to construct ``PartitionSpec`` / ``NamedSharding``
    themselves.
    """

    mesh: Mesh
    data_partition: PartitionSpec
    param_partition: PartitionSpec
    data_sharding: NamedSharding
    param_sharding: NamedSharding


class JaxDeviceManager:
    """Manage JAX devices, environment, mesh, and sharding.

    Analogous to :class:`~blacksmith.tools.device_manager.DeviceManager`
    (Torch) but for JAX/EasyDel experiments.

    Expects the config to carry the standard fields from
    :class:`~blacksmith.tools.templates.configs.TrainingConfig` plus
    optional JAX-specific attributes (``num_devices``,
    ``apply_gqa_workaround``, ``optimizer_on_cpu``, etc.).
    Missing attributes fall back to safe defaults via ``getattr``.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

        self._setup_env()

        self.device, self.device_kind = self._select_device()
        jax.config.update("jax_default_device", self.device)

        if getattr(config, "apply_gqa_workaround", True) and self.device_kind == "tt":
            apply_gqa_workaround()

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
            os.environ.setdefault("CONVERT_SHLO_TO_SHARDY", "1")
            os.environ.setdefault("DISABLE_NUMERIC_CC_TOKEN", "1")
            jax.config.update("jax_use_shardy_partitioner", True)

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
        """Build a :class:`jax.sharding.Mesh`.

        Defaults to a single-axis ``("data",)`` mesh when explicit
        ``mesh_shape`` / ``mesh_axis_names`` are not provided.
        """
        n = getattr(self.config, "num_devices", 1)
        shape = tuple(getattr(self.config, "mesh_shape", None) or [n])
        names = tuple(getattr(self.config, "mesh_axis_names", None) or ["data"])

        if self.device_kind == "tt":
            devices = tuple(jax.devices("tt")[:n])
        elif self.device_kind == "gpu":
            devices = tuple(jax.devices("gpu")[:n])
        else:
            devices = tuple(jax.devices("cpu")[:n])

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
        """Replicate a pytree across the mesh (no sharding)."""
        return jax.tree.map(
            lambda x: jax.device_put(x, self.sharding.param_sharding),
            pytree,
        )

    @staticmethod
    def to_cpu(pytree):
        """Move every leaf of *pytree* to CPU."""
        cpu = jax.devices("cpu")[0]
        return jax.tree.map(lambda x: jax.device_put(x, cpu), pytree)

    def to_device(self, pytree):
        """Move every leaf onto ``self.device``."""
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
        """Apply an optax update, using the CPU workaround when needed.

        When ``optimizer_on_cpu`` is True and the device is TT,
        params/grads/opt_state are moved to CPU before the update and
        back to device afterwards (workaround for ``tt-metal#27072``).
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
        """Axis size to pass to EasyDel ``sharding_axis_dims``.

        On TT this is the *total* number of TT devices visible to
        JAX (not necessarily the mesh size), because EasyDel uses
        this at model-load time to decide internal sharding.
        """
        if self.device_kind == "tt":
            try:
                return len(jax.devices("tt"))
            except Exception:
                return 1
        return 1

    def describe(self) -> dict:
        """Summary dict suitable for ``TrainingLogger.log_model_info``."""
        return {
            "device": self.device_kind,
            "num_devices": getattr(self.config, "num_devices", 1),
            "mesh_shape": list(self.mesh.shape.values()),
            "mesh_axis_names": list(self.mesh.shape.keys()),
            "data_parallel": self.is_data_parallel(),
            "optimizer_on_cpu": (getattr(self.config, "optimizer_on_cpu", True) and self.device_kind == "tt"),
            "gqa_workaround": (
                getattr(
                    self.config,
                    "apply_gqa_workaround",
                    True,
                )
                and self.device_kind == "tt"
            ),
        }

    def __repr__(self) -> str:
        return (
            f"JaxDeviceManager("
            f"device={self.device_kind!r}, "
            f"num_devices={getattr(self.config, 'num_devices', 1)}, "
            f"mesh={dict(self.mesh.shape)})"
        )
