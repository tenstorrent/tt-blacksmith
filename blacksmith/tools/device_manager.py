# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import re
from typing import Dict, List, Optional, Tuple

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
        if hasattr(self.config, "mesh_shape") and self.config.mesh_shape:
            os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
            os.environ["MESH_SHAPE"] = ",".join(map(str, self.config.mesh_shape))
            os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
            os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
            xr.use_spmd()

    def _create_mesh(self) -> Optional[xs.Mesh]:
        # Check if mesh configuration is provided.
        if not hasattr(self.config, "mesh_shape") or not self.config.mesh_shape:
            return None

        if not hasattr(self.config, "mesh_axis_names") or not self.config.mesh_axis_names:
            return None

        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))

        # Read mesh_shape from config.
        mesh_shape = tuple(self.config.mesh_shape)

        # Read mesh axis names from config.
        axis_names = tuple(self.config.mesh_axis_names)

        return xs.Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)

    def is_data_parallel(self) -> bool:
        """Check if data parallelism is enabled based on mesh configuration."""
        return self.mesh is not None and "data" in self.mesh.axis_names and self.mesh.shape()["data"] > 1

    def is_tensor_parallel(self) -> bool:
        """Check if tensor parallelism is enabled based on mesh configuration."""
        return self.mesh is not None and "model" in self.mesh.axis_names and self.mesh.shape()["model"] > 1

    def shard_tensor(self, tensor: torch.Tensor, sharding_spec: Tuple):
        return xs.mark_sharding(tensor, self.mesh, sharding_spec)

    def shard_model(self, model: nn.Module) -> nn.Module:
        """Shard model based on mesh configuration."""
        if self.is_tensor_parallel():
            return self._apply_tensor_parallelism(model)

        return model

    def _apply_tensor_parallelism(self, model: nn.Module) -> nn.Module:
        """Apply tensor parallelism using regex pattern matching from config."""
        torch_xla.sync(wait=True)

        # Get sharding patterns from config (list of [pattern, spec] pairs).
        sharding_patterns = getattr(self.config, "model_sharding_patterns", None)
        assert sharding_patterns is not None, "model_sharding_patterns must be provided for tensor parallelism"

        # Use regex pattern matching on named_modules.
        for name, module in model.named_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue

            for pattern_spec in sharding_patterns:
                pattern = pattern_spec[0]
                shard_spec = tuple(pattern_spec[1])

                if re.search(pattern, name):
                    xs.mark_sharding(module.weight, self.mesh, shard_spec)
                    break  # Stop after first match.

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
                    partition_spec = ("data",) + tuple([None] * (tensor.dim() - 1))
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
