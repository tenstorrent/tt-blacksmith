# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import contextlib
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from blacksmith.configs.trainer import TrainerConfig


class DeviceManager(ABC):
    """Framework-agnostic view of the device / mesh configuration."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        # Set by concrete subclasses once the device / mesh are opened.
        self.mesh: Optional[Any] = None
        self.device: Optional[Any] = None
        self._validate_mesh_config()

    # ------------------------------------------------------------------ mesh
    def _validate_mesh_config(self) -> None:
        """Validate the mesh configuration (no device access, no framework imports)."""
        if not getattr(self.config, "mesh_shape", None):
            return

        shape = tuple(self.config.mesh_shape)
        axis_names = self.config.mesh_axis_names
        assert axis_names is not None, "Mesh axis names must be provided for multichip parallelism."
        assert len(shape) == len(axis_names), "Mesh shape and axis names must have the same length."
        assert (self.config.input_sharding_dim is None) or (
            self.config.input_sharding_dim in axis_names
        ), "`input_sharding_dim` must be None or it should be present in `mesh_axis_names`."
        for pattern_spec in self.config.model_sharding_patterns or []:
            for dimension in pattern_spec[1]:
                if dimension is not None:
                    assert (
                        dimension in axis_names and shape[axis_names.index(dimension)] > 1
                    ), f"Dimension {dimension} is not present in `mesh_axis_names` or it has size 1 for model sharding pattern {pattern_spec}."

    def _axis_index(self, axis: str) -> int:
        return self.config.mesh_axis_names.index(axis)

    # -------------------------------------------------------------- queries
    def is_data_parallel(self) -> bool:
        """Check if data parallelism is enabled based on mesh configuration."""
        return (
            self.config.input_sharding_dim is not None
            and self.mesh is not None
            and self.config.mesh_shape[self._axis_index(self.config.input_sharding_dim)] > 1
        )

    def is_tensor_parallel(self) -> bool:
        """Check if tensor parallelism is enabled based on mesh configuration."""
        return self.config.model_sharding_patterns is not None and self.mesh is not None

    # --------------------------------------------------------------- abstract
    @abstractmethod
    def shard_model(self, model: Any) -> Any:
        """Shard the model according to the mesh configuration."""
        pass

    @abstractmethod
    def prepare_batch(self, batch: Dict[str, Any], skip_keys: tuple[str, ...] = ()) -> Dict[str, Any]:
        """Move the batch to device and apply data-parallel sharding if configured."""
        pass

    @abstractmethod
    def optimizer_step(self, optimizer: Any, zero_grad: bool = False) -> None:
        """Perform an optimizer step, optionally zeroing gradients afterwards."""
        pass

    # --------------------------------------------------------------- context
    def replication_context(self):
        """Context in which plain tensors count as replicated. No-op by default."""
        return contextlib.nullcontext()
