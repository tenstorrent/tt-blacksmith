# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

from blacksmith.tools.templates.configs import TrainingConfig


class DeviceManager:
    """Manages different parallelization strategies based on mesh configuration."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.mesh = None

        self._setup()

    def _setup(self):
        if not self.config.use_tt:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return

        self._setup_tt_environment()
        self.device = torch_xla.device()

        self.mesh = self._create_mesh()

    def _setup_tt_environment(self):
        # Setup for single device.
        xr.set_device_type("TT")
        os.environ["PJRT_DEVICE"] = "TT"
        os.environ["XLA_STABLEHLO_COMPILE"] = "1"

        # Additional setup for multichip (if mesh configuration is provided).
        if hasattr(self.config, "mesh_shape") and self.config.mesh_shape is not None:
            os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
            os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
            os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
            xr.use_spmd()

    def _create_mesh(self) -> Optional[xs.Mesh]:
        # Check if mesh configuration is provided.
        if not hasattr(self.config, "mesh_shape") or not self.config.mesh_shape:
            return None

        # Check if mesh configuration is valid.
        assert self.config.mesh_axis_names is not None, "Mesh axis names must be provided for multichip parallelism."
        assert (self.config.input_sharding_dim is None) or (
            self.config.input_sharding_dim in self.config.mesh_axis_names
        ), "`input_sharding_dim` must be None or it should be present in `mesh_axis_names`."
        if self.config.model_sharding_patterns is not None:
            for pattern_spec in self.config.model_sharding_patterns:
                dimensions = pattern_spec[1]
                for dimension in dimensions:
                    if dimension is not None:
                        assert (
                            dimension in self.config.mesh_axis_names
                            and self.config.mesh_shape[self.config.mesh_axis_names.index(dimension)] > 1
                        ), f"Dimension {dimension} is not present in `mesh_axis_names` or it has size 1 for model sharding pattern {pattern_spec}."

        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        assert len(self.config.mesh_shape) == len(
            self.config.mesh_axis_names
        ), "Mesh shape and axis names must have the same length."

        return xs.Mesh(
            device_ids=device_ids,
            mesh_shape=tuple(self.config.mesh_shape),
            axis_names=tuple(self.config.mesh_axis_names),
        )

    def is_data_parallel(self) -> bool:
        """Check if data parallelism is enabled based on mesh configuration."""

        return (
            self.config.input_sharding_dim is not None
            and self.mesh is not None
            and self.mesh.shape()[self.config.input_sharding_dim] > 1
        )

    def is_tensor_parallel(self) -> bool:
        """Check if tensor parallelism is enabled based on mesh configuration."""
        return self.config.model_sharding_patterns is not None and self.mesh is not None

    def shard_tensor(self, tensor: torch.Tensor, sharding_spec: Tuple):
        return xs.mark_sharding(tensor, self.mesh, sharding_spec)

    def shard_model(self, model: nn.Module) -> nn.Module:
        """Shard model based on mesh configuration."""
        if self.is_tensor_parallel():
            return self._apply_tensor_parallelism(model)

        return model

    def _apply_tensor_parallelism(self, model: nn.Module) -> nn.Module:
        """Apply tensor parallelism using regex pattern matching from config."""
        sharding_patterns = self.config.model_sharding_patterns

        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue
            match = next((ps for ps in sharding_patterns if re.search(ps[0], name)), None)
            if match and torch_xla._XLAC._get_xla_sharding_spec(module.weight) in (None, ""):
                xs.mark_sharding(module.weight, self.mesh, tuple(match[1]))

        # Shard parameters by name (for nn.Parameter, biases, etc. not reachable via module.weight).
        param_patterns = getattr(self.config, "param_sharding_patterns", [])
        for name, param in model.named_parameters():
            match = next((ps for ps in param_patterns if re.search(ps[0], name)), None)
            if match and torch_xla._XLAC._get_xla_sharding_spec(param) in (None, ""):
                xs.mark_sharding(param, self.mesh, tuple(match[1]))

        torch_xla.sync(wait=True)
        return model

    def shard_optimizer(self, optimizer: torch.optim.Optimizer):
        raise NotImplementedError("Optimizer sharding is not implemented yet.")

    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training, applying data parallel sharding if configured."""
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.is_data_parallel():
            for _, tensor in batch.items():
                if tensor.dim() > 0:
                    partition_spec = (self.config.input_sharding_dim,) + tuple([None] * (tensor.dim() - 1))
                    xs.mark_sharding(tensor, self.mesh, partition_spec)

        return batch

    def optimizer_step(self, optimizer: torch.optim.Optimizer):
        """Perform optimizer step with appropriate synchronization."""
        if self.mesh is None:
            # Single device
            optimizer.step()
            if self.config.use_tt:
                torch_xla.sync(wait=True)
        else:
            # For multichip - xm.optimizer_step forces execution and ensures correct all-reduce operations
            xm.optimizer_step(optimizer, barrier=True)
