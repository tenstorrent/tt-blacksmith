# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Optional, Tuple, Dict
from enum import Enum

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import numpy as np

from blacksmith.tools.templates.configs import TrainingConfig


class ParallelStrategy(Enum):
    SINGLE = "single"
    DATA_PARALLEL = "data_parallel"
    TENSOR_PARALLEL = "tensor_parallel"


class DeviceManager:
    """Manages different parallelization strategies."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.strategy = config.parallelism_strategy

        self._setup()

    def _setup(self):
        if not self.config.use_tt:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return

        self._setup_tt_environment()
        self.device = torch_xla.device()

        self.mesh = self._create_mesh()

    def _setup_tt_environment(self):
        # Setup for single device
        xr.set_device_type("TT")
        os.environ["PJRT_DEVICE"] = "TT"
        os.environ["XLA_STABLEHLO_COMPILE"] = "1"

        # Additional setup for multichip
        if self.strategy != ParallelStrategy.SINGLE:
            os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
            os.environ["MESH_SHAPE"] = self.config.mesh_shape
            os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
            os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
            xr.use_spmd()

    def _create_mesh(self) -> Optional[xs.Mesh]:
        if self.strategy == ParallelStrategy.SINGLE:
            return None

        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))
        mesh_shape = None
        axis_names = None

        if self.strategy == ParallelStrategy.DATA_PARALLEL:
            mesh_shape = (num_devices, 1)
            axis_names = ("data", "model")
        elif self.strategy == ParallelStrategy.TENSOR_PARALLEL:
            mesh_shape = (1, num_devices)
            axis_names = ("data", "model")
        else:
            supported_strategies = [f for f in ParallelStrategy]
            raise ValueError(f"Invalid parallelism: {self.strategy}. Supported strategies: {supported_strategies}.")

        return xs.Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)

    def shard_tensor(self, tensor: torch.Tensor, sharding_spec: Tuple):
        return xs.mark_sharding(tensor, self.mesh, sharding_spec)

    def shard_model(self, model: nn.Module) -> nn.Module:
        if self.strategy == ParallelStrategy.TENSOR_PARALLEL:
            return self._apply_tensor_parallelism(model)

        return model

    def _apply_tensor_parallelism(self, model: nn.Module) -> nn.Module:
        torch_xla.sync(wait=True)

        sharding_specs = self.config.tp_sharding_specs or {}
        for name, param in model.named_parameters():
            if param.dim() == 0:
                continue

            partition_spec = sharding_specs.get(name, None)
            if partition_spec is not None:
                xs.mark_sharding(param, self.mesh, partition_spec)

        return model

    def shard_optimizer(self, optimizer: torch.optim.Optimizer):
        raise NotImplementedError("Optimizer sharding is not implemented yet.")

    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.strategy == ParallelStrategy.DATA_PARALLEL:
            for _, tensor in batch.items():
                if tensor.dim() > 0:
                    partition_spec = ("data",) + tuple([None] * (tensor.dim() - 1))
                    xs.mark_sharding(tensor, self.mesh, partition_spec)

        return batch

    def optimizer_step(self, optimizer: torch.optim.Optimizer):
        if self.strategy == ParallelStrategy.SINGLE:
            optimizer.step()
            if self.config.use_tt:
                torch_xla.sync(wait=True)
        else:
            # For multichip - xm.optimizer_step forces execution and ensures correct all-reduce operations
            xm.optimizer_step(optimizer, barrier=True)
