# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import contextlib
import math
import os
import re
import warnings
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
    Placement,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)

from blacksmith.configs.training import TrainingConfig


def tt_compile_options(config) -> dict:
    """tt-crank options for `torch.compile(fn, backend="tt", options=...)`."""
    from tt_crank.torch._compile import BfpDtype, CompileOption, MathFidelity

    options = {
        CompileOption.FP32_DEST_ACC_EN: config.fp32_dest_acc_en,
        CompileOption.MATH_FIDELITY: getattr(MathFidelity, config.math_fidelity),
        CompileOption.OPT_LEVEL: config.optimization_level,
        CompileOption.ENABLE_CONST_EVAL: config.enable_const_eval,
        CompileOption.ENABLE_TRACE: config.enable_trace,
    }
    weight_dtype = config.experimental_weight_dtype
    if weight_dtype and weight_dtype != "bf16":
        bfp = {"bfp_bf8": "BfpBf8", "bfp_bf4": "BfpBf4"}.get(weight_dtype.lower(), weight_dtype)
        options[CompileOption.EXPERIMENTAL_WEIGHT_DTYPE] = getattr(BfpDtype, bfp)
    return options


class DeviceManager:
    """Owns the `tt` device and, for multichip runs, the DTensor mesh."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.mesh: Optional[DeviceMesh] = None

        if not config.use_tt:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return

        # Importing the package registers the "tt" PrivateUse1 device, the "tt"
        # dynamo backend and the "tt" c10d backend. There is no PJRT_DEVICE /
        # XLA_* environment to set.
        import tt_crank.torch  # noqa: F401

        self.device = torch.device("tt")
        self.mesh = self._create_mesh()

    # ------------------------------------------------------------------ mesh
    def _create_mesh(self) -> Optional[DeviceMesh]:
        if not getattr(self.config, "mesh_shape", None):
            return None

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
        if "fsdp" in axis_names:
            warnings.warn(
                "mesh axis 'fsdp' requested but FSDP is not ported to tt-crank; "
                "parameters along that axis are replicated (memory use matches DP, not FSDP).",
                stacklevel=2,
            )

        available = torch.tt.num_chips()
        requested = math.prod(shape)
        assert requested <= available, f"mesh {shape} needs {requested} chips, only {available} available"

        self._init_process_group(requested)
        # init_device_mesh opens the runtime MeshDevice at this shape *and* builds
        # the torch DeviceMesh; the two must not be set up separately.
        return torch.tt.init_device_mesh(shape, mesh_dim_names=tuple(axis_names))

    @staticmethod
    def _init_process_group(world_size: int) -> None:
        """Bring up the "tt" c10d process group that DTensor collectives use."""
        import torch.distributed as dist
        from torch.testing._internal.distributed.fake_pg import FakeStore

        if dist.is_initialized():
            return
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        dist.init_process_group(backend="tt", rank=0, world_size=world_size, store=FakeStore())

    def _axis_index(self, axis: str) -> int:
        return self.config.mesh_axis_names.index(axis)

    def _placements(self, sharding_spec: Sequence[Optional[str]]) -> list[Placement]:
        """tt-xla partition spec -> DTensor placements."""
        placements: list[Placement] = [Replicate() for _ in self.config.mesh_shape]
        for dim, axis in enumerate(sharding_spec):
            if axis is not None:
                placements[self._axis_index(axis)] = Shard(dim)
        return placements

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

    def is_fsdp(self) -> bool:
        """Check if FSDP is requested by the mesh configuration (not ported; see module docstring)."""
        return self.mesh is not None and "fsdp" in self.config.mesh_axis_names

    # ------------------------------------------------------------- sharding
    def shard_tensor(self, tensor: torch.Tensor, sharding_spec: Sequence[Optional[str]]) -> torch.Tensor:
        if self.mesh is None:
            return tensor
        return distribute_tensor(tensor, self.mesh, self._placements(sharding_spec))

    def shard_model(self, model: nn.Module) -> nn.Module:
        """Shard model based on mesh configuration."""
        if self.mesh is None or self._is_distributed(model):
            return model
        return distribute_module(model, self.mesh, partition_fn=self._partition_fn(model))

    @staticmethod
    def _is_distributed(model: nn.Module) -> bool:
        return isinstance(next(model.parameters(), None), DTensor)

    def _partition_fn(self, model: nn.Module):
        module_patterns = [(re.compile(p), tuple(s)) for p, s in (self.config.model_sharding_patterns or [])]
        param_patterns = [(re.compile(p), tuple(s)) for p, s in getattr(self.config, "param_sharding_patterns", [])]
        replicate = [Replicate() for _ in self.config.mesh_shape]

        # Resolve the per-parameter placement up front, by the same names the tt-xla
        # DeviceManager matched on: module name for `.weight`, full parameter name
        # for `param_sharding_patterns`.
        placements_by_param: dict[str, list[Placement]] = {}
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, nn.Parameter):
                continue
            match = next((spec for pattern, spec in module_patterns if pattern.search(name)), None)
            if match is not None:
                placements_by_param[f"{name}.weight" if name else "weight"] = self._placements(match)
        for name, _ in model.named_parameters():
            match = next((spec for pattern, spec in param_patterns if pattern.search(name)), None)
            if match is not None:
                placements_by_param.setdefault(name, self._placements(match))

        def partition_fn(name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
            # Place every direct parameter ourselves, preserving requires_grad: torch's own
            # replicate pass in `distribute_module` rebuilds untouched parameters as bare
            # `nn.Parameter(...)`, i.e. trainable, which under LoRA would unfreeze the base model.
            for param_name, param in list(module._parameters.items()):
                if param is None or isinstance(param, DTensor):
                    continue
                full_name = f"{name}.{param_name}" if name else param_name
                placements = placements_by_param.get(full_name, replicate)
                module.register_parameter(
                    param_name,
                    nn.Parameter(distribute_tensor(param, device_mesh, placements), requires_grad=param.requires_grad),
                )

        return partition_fn

    def shard_optimizer(self, optimizer: torch.optim.Optimizer):
        raise NotImplementedError("Optimizer sharding is not implemented yet.")

    # ----------------------------------------------------------------- step
    def prepare_batch(
        self,
        batch: Dict[str, torch.Tensor],
        skip_keys: tuple[str, ...] = (),
    ) -> Dict[str, torch.Tensor]:
        """Move the batch to device and apply data-parallel sharding if configured.

        ``skip_keys`` stay on the host.
        """
        batch = {k: v if k in skip_keys else v.to(self.device) for k, v in batch.items()}

        if self.is_data_parallel():
            for key, tensor in batch.items():
                if key in skip_keys or tensor.dim() == 0:
                    continue
                partition_spec = (self.config.input_sharding_dim,) + (None,) * (tensor.dim() - 1)
                batch[key] = distribute_tensor(tensor, self.mesh, self._placements(partition_spec))

        return batch

    def optimizer_step(self, optimizer: torch.optim.Optimizer, zero_grad: bool = False) -> None:
        """
        Perform optimizer step, optionally zeroing grads afterwards.

        Under data parallelism the gradient all-reduce is implicit (``Partial`` DTensor grads).
        """
        optimizer.step()
        if zero_grad:
            optimizer.zero_grad(set_to_none=True)

    def compile_options(self) -> dict:
        """See `tt_compile_options`."""
        return tt_compile_options(self.config)

    def replication_context(self):
        """Context in which plain tensors HF builds internally count as replicated DTensors."""
        if self.mesh is None:
            return contextlib.nullcontext()

        from torch.distributed.tensor.experimental import implicit_replication

        return implicit_replication()
