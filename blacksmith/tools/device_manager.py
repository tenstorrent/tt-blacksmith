# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import re
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
        if self.strategy != ParallelStrategy.SINGLE.value:
            os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
            os.environ["MESH_SHAPE"] = self.config.mesh_shape
            os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
            os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"
            xr.use_spmd()

    def _create_mesh(self) -> Optional[xs.Mesh]:
        if self.strategy == ParallelStrategy.SINGLE.value:
            return None

        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))
        mesh_shape = None
        axis_names = None

        if self.strategy == ParallelStrategy.DATA_PARALLEL.value:
            mesh_shape = (num_devices, 1)
            axis_names = ("data", "model")
        elif self.strategy == ParallelStrategy.TENSOR_PARALLEL.value:
            mesh_shape = (2, num_devices // 2)
            axis_names = ("data", "model")
        else:
            supported_strategies = [f.value for f in ParallelStrategy]
            raise ValueError(f"Invalid parallelism: {self.strategy}. Supported strategies: {supported_strategies}.")

        return xs.Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)

    def is_data_parallel(self) -> bool:
        return (
            self.mesh is not None
            and "data" in self.mesh.axis_names
            and self.mesh.shape()["data"] > 1
        )

    def is_tensor_parallel(self) -> bool:
        return (
            self.mesh is not None
            and "model" in self.mesh.axis_names
            and self.mesh.shape()["model"] > 1
        )

    def shard_tensor(self, tensor: torch.Tensor, sharding_spec: Tuple):
        return xs.mark_sharding(tensor, self.mesh, sharding_spec)

    def shard_model(self, model: nn.Module) -> nn.Module:
        if self.is_tensor_parallel():
            print(f"[DeviceManager] Applying tensor parallelism to the model...", flush =True)
            return self._apply_tensor_parallelism(model)

        return model

    def _apply_tensor_parallelism(self, model: nn.Module) -> nn.Module:
        torch_xla.sync(wait=True)

        # Regex → sharding pattern
        rules = [
            # === Attention ===
            (r"\.self_attn\.q_proj\.base_layer$",      ("model", None)),
            (r"\.self_attn\.q_proj\.lora_B\.default$", ("model", None)),

            (r"\.self_attn\.k_proj$",                  ("model", None)),

            (r"\.self_attn\.v_proj\.base_layer$",      ("model", None)),
            (r"\.self_attn\.v_proj\.lora_B\.default$", ("model", None)),

            (r"\.self_attn\.o_proj$",                  (None, "model")),

            # === MLP ===
            (r"\.mlp\.gate_proj$", ("model", None)),
            (r"\.mlp\.up_proj$",   ("model", None)),
            (r"\.mlp\.down_proj$", (None, "model")),
        ]

        # Iterate and match
        for name, module in model.named_modules():
            print(f"[TP] Checking module: {name}", flush=True)
            if not hasattr(module, "weight") or module.weight is None:
                continue

            for pattern, shard in rules:
                if re.search(pattern, name):
                    xs.mark_sharding(module.weight, self.mesh, shard)
                    #print(f"[TP] {name}.weight → {shard}", flush=True)
                    print(f"Sharded {name}.weight with spec {shard}", flush=True)
                    break  # stop after first match
        
        torch_xla.sync(wait=True)

    def shard_optimizer(self, optimizer: torch.optim.Optimizer):
        raise NotImplementedError("Optimizer sharding is not implemented yet.")

    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.is_data_parallel():
            for _, tensor in batch.items():
                if tensor.dim() > 0:
                    partition_spec = ("data",) + tuple([None] * (tensor.dim() - 1))
                    xs.mark_sharding(tensor, self.mesh, partition_spec)

        return batch

    def optimizer_step(self, optimizer: torch.optim.Optimizer):
        if self.strategy == ParallelStrategy.SINGLE.value:
            optimizer.step()
            if self.config.use_tt:
                torch_xla.sync(wait=True)
        else:
            # For multichip - xm.optimizer_step forces execution and ensures correct all-reduce operations
            xm.optimizer_step(optimizer, barrier=True)
