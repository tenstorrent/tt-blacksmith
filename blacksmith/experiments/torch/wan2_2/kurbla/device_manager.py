# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
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
from typing import cast

import tt_kurbla.torch  # noqa: F401  — registers the "tt" device, c10d backend, dynamo backend, torch.tt
from tt_kurbla.torch._compile import CompileOption, MathFidelity

from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig

# Compile options for the `tt` dynamo backend. Passed as a raw dict — `tt_backend`
# converts it via `_compile_options` itself, so pre-converting here would double-convert.
_COMPILE_OPTIONS = {
    CompileOption.FP32_DEST_ACC_EN: True,
    CompileOption.MATH_FIDELITY: MathFidelity.HiFi4,
    CompileOption.EXPERIMENTAL_ENABLE_DRAM_SPACE_SAVING_OPTIMIZATION: True,
}

def _patch_dtensor_conv() -> None:
    import torch.distributed.tensor._tp_conv as _tp_conv
    from torch.distributed.tensor import DTensor

    def _is_supported_patched(input_size, kernel_size, stride, padding, dilation):
        return True

    _tp_conv._is_supported = _is_supported_patched
    import torch.nn.functional as F

    _orig_pad = F.pad

    @torch.jit.ignore
    def _pad_maybe_dtensor(input, pad, mode: str, value) -> torch.Tensor:
        if isinstance(input, DTensor):
            mesh, placements = input.device_mesh, input.placements
            local = input.to_local()
            out_local = _orig_pad(local, pad, mode=mode, value=value)
            return DTensor.from_local(out_local, device_mesh=mesh, placements=placements)
        return _orig_pad(input, pad, mode=mode, value=value)

    def _pad_dtensor_safe(
        input: torch.Tensor, pad: List[int], mode: str = 'constant', value: Optional[float] = None
    ) -> torch.Tensor:
        if not torch.jit.is_scripting():
            return _pad_maybe_dtensor(input, pad, mode, value)
        return _orig_pad(input, pad, mode=mode, value=value)

    F.pad = _pad_dtensor_safe


class WanDeviceManager:

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.mesh: Optional[DeviceMesh] = None
        self._compile_cache: dict = {}
        self.compile_options: dict = dict(_COMPILE_OPTIONS) | {
            CompileOption.OPT_LEVEL: config.optimization_level,
        }

        self._setup()

    def _setup(self) -> None:
        if not self.config.use_tt:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return

        self.device = torch.device("tt")
        self.mesh = self._create_mesh()

    def _create_mesh(self) -> Optional[DeviceMesh]:
        if not self.config.mesh_shape:
            return None

        assert self.config.mesh_axis_names is not None, "Mesh axis names must be provided for multichip parallelism."
        assert len(self.config.mesh_shape) == len(
            self.config.mesh_axis_names
        ), "Mesh shape and axis names must have the same length."
        assert (self.config.input_sharding_dim is None) or (
            self.config.input_sharding_dim in self.config.mesh_axis_names
        ), "`input_sharding_dim` must be None or it should be present in `mesh_axis_names`."

        mesh_size = math.prod(self.config.mesh_shape)
        num_chips = torch.tt.num_chips()
        assert mesh_size <= num_chips, f"Mesh {tuple(self.config.mesh_shape)} needs {mesh_size} chips, found {num_chips}."

        if not dist.is_initialized():
            dist.init_process_group(backend="tt", rank=0, world_size=mesh_size, store=dist.HashStore())

        _patch_dtensor_conv()

        return torch.tt.init_device_mesh(
            tuple(self.config.mesh_shape), mesh_dim_names=tuple(self.config.mesh_axis_names)
        )

    def is_data_parallel(self) -> bool:
        """Check if data parallelism is enabled based on mesh configuration."""
        if self.config.input_sharding_dim is None or self.mesh is None:
            return False
        return self.mesh.size(self.mesh.mesh_dim_names.index(self.config.input_sharding_dim)) > 1

    def is_tensor_parallel(self) -> bool:
        """Check if tensor parallelism is enabled based on mesh configuration."""
        return self.config.model_sharding_patterns is not None and self.mesh is not None

    def to_device(self, module_or_tensor):
        """Move a module or tensor to the tt device.

        With a mesh active, tensors also become replicated `DTensor`s: `shard_model`
        makes every parameter a DTensor, and an op mixing a plain tensor with a DTensor
        raises. Replication is what `.to("tt")` already does physically (one handle,
        same data on every chip) — this only makes it visible to DTensor.
        """
        out = module_or_tensor.to(self.device)
        if self.mesh is not None and isinstance(out, torch.Tensor):
            out = self._replicate(out)
        return out

    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Apply the tt-kurbla graph rewrites to a freshly constructed model.

        Counterpart of the no-op hook on the tt-xla manager: the VAE's `WanCausalConv3d`
        and `ZeroPad2d` modules have to be retyped per instance, which class-level
        patches cannot do. Idempotent, so re-running it on an already-rewritten model is
        harmless.
        """
        # from blacksmith.experiments.torch.wan2_2.kurbla.model_overrides import apply_kurbla_overrides

        # apply_kurbla_overrides(model)
        return model

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Full, mesh-wide plain tensor from a (possibly distributed) one.

        The inverse of `to_device`/`prepare_batch`, for values leaving the device:
        `full_tensor()` fires whatever collective the placement needs (all-gather for
        `Shard`, all-reduce for `Partial`). A no-op on plain tensors.
        """
        return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor

    def shard_tensor(self, tensor: torch.Tensor, sharding_spec: Sequence[Optional[str]]) -> DTensor:
        """Distribute `tensor` per a partition spec (one mesh axis name, or None, per
        tensor dim) — the DTensor counterpart of the base manager's `mark_sharding`."""
        return distribute_tensor(tensor, self.mesh, self._placements(sharding_spec, tensor))

    def shard_model(self, model: nn.Module) -> nn.Module:
        """Distribute every parameter and buffer of `model` over the mesh.

        Params named by `model_sharding_patterns` (matched on module name, applied to
        `module.weight`) or `param_sharding_patterns` (matched on the full parameter
        name) get that pattern's placements; everything else is replicated. DTensor
        conversion is all-or-nothing — a single plain parameter left behind makes the
        first op touching it raise on mixed operands — so unmatched tensors are
        replicated rather than skipped, which is exactly what the XLA path got
        implicitly from GSPMD inference.
        """
        if self.mesh is None:
            return model

        module_patterns = self.config.model_sharding_patterns or []
        param_patterns = self.config.param_sharding_patterns or []
        full_replicate = [Replicate()] * self.mesh.ndim

        def partition_fn(name: str, module: nn.Module, device_mesh) -> None:
            # Module patterns address `module.weight`; a param pattern can name any
            # parameter. Module patterns win on `weight`, matching the base manager,
            # where the param pass skipped tensors the module pass had annotated.
            weight_spec = _match(module_patterns, name)

            for param_name, param in list(module.named_parameters(recurse=False)):
                if isinstance(param, DTensor):  # already distributed (shard_model re-run)
                    continue
                qualified_name = f"{name}.{param_name}" if name else param_name
                spec = weight_spec if param_name == "weight" else None
                if spec is None:
                    spec = _match(param_patterns, qualified_name)
                placements = full_replicate if spec is None else self._placements(spec, param, qualified_name)
                # Replicate here too rather than leaving it to `distribute_module`: its
                # fallback rebuilds the Parameter with the default requires_grad=True,
                # which would un-freeze the LoRA base weights.
                print(f"[shard_model] {qualified_name}: {param.shape} -> {placements}")
                _distribute_param(module, param_name, device_mesh, placements)

        # Buffers (unmatched by construction — the patterns name parameters) are
        # replicated by `distribute_module` itself.
        return distribute_module(model.to(self.device), self.mesh, partition_fn=partition_fn)

    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch for training, applying data parallel sharding if configured."""
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.mesh is None:
            return batch

        shard_batch_dim = self.is_data_parallel()
        prepared = {}
        for name, tensor in batch.items():
            if shard_batch_dim and tensor.dim() > 0:
                spec = [self.config.input_sharding_dim] + [None] * (tensor.dim() - 1)
                prepared[name] = self.shard_tensor(tensor, spec)
            else:
                prepared[name] = self._replicate(tensor)
        return prepared

    def compile(self, module: nn.Module):
        """Cached on id(module), like the tt-xla manager: callers must keep the wrapper
        alive across calls.

        `generate_validation_sample` calls this on the transformer at every validation,
        so without the cache each one built a fresh `OptimizedModule` around a module
        that was already compiled. `self._compile_cache` existed for this and was simply
        never consulted.
        """
        cached = self._compile_cache.get(id(module))
        if cached is None:
            print(f"[compile] compiling {type(module).__name__} with {self.compile_options}", flush=True)
            cached = torch.compile(module, backend="tt", options=self.compile_options)
            self._compile_cache[id(module)] = cached
        else:
            print(f"[compile] reusing the compiled {type(module).__name__}", flush=True)
        return cached

    def optimizer_step(self, optimizer: torch.optim.Optimizer, zero_grad: bool = False):
        """Perform the optimizer step, as one compiled graph when `compile_optimizer` is set.

        No explicit all-reduce: with DTensor parameters the gradients are DTensors too,
        and a data-parallel gradient comes out `Partial` over the batch axis. The step
        combines it with the replicated parameter, so DTensor emits the reduction itself.

        Run eagerly, the step is one program submit per parameter per op. Dynamo traces
        the whole of AdamW's single-tensor path without a graph break, so compiling
        `optimizer.step` turns those into one graph. Cached on id(optimizer) like
        `compile`, because a fresh wrapper each call would defeat the point.
        """
        if not getattr(self.config, "compile_optimizer", False):
            optimizer.step()
            if zero_grad:
                optimizer.zero_grad(set_to_none=False)
            return

        step = self._compile_cache.get(id(optimizer))
        if step is None:
            print(f"[compile] compiling the {type(optimizer).__name__} step", flush=True)
            step = torch.compile(optimizer.step, backend="tt", options=self.compile_options)
            self._compile_cache[id(optimizer)] = step
        step()
        if zero_grad:
            # Kept as tensors (not None) so the next window's grads accumulate.
            optimizer.zero_grad(set_to_none=False)

    def sync(self) -> None:
        """No-op: tt_kurbla executes eagerly, there is no lazy graph to flush.

        Kept so the experiment scripts' `device_manager.sync()` calls (needed on the
        XLA path) stay valid here.
        """
        return

    def _replicate(self, tensor: torch.Tensor) -> DTensor:
        if isinstance(tensor, DTensor):
            return tensor
        return distribute_tensor(tensor, self.mesh, [Replicate()] * self.mesh.ndim)

    def _placements(
        self, spec: Sequence[Optional[str]], tensor: torch.Tensor, name: str = "tensor"
    ) -> List[Placement]:
        """Translate a partition spec into DTensor placements.

        The YAML specs are XLA/GSPMD partition specs — one entry *per tensor dim*, naming
        the mesh axis that splits it (or None to leave it whole). DTensor is indexed the
        other way round: one placement *per mesh dim*, naming the tensor dim that axis
        splits. So `["model", "batch"]` on a 2-D weight over a ("batch", "model") mesh
        becomes `[Shard(1), Shard(0)]` — mesh axis "batch" splits tensor dim 1, "model"
        splits dim 0.
        """
        axis_names = self.mesh.mesh_dim_names
        unknown = {axis for axis in spec if axis is not None} - set(axis_names)
        assert not unknown, f"{name}: sharding spec {list(spec)} names axes {sorted(unknown)} absent from mesh {axis_names}."
        assert len(spec) == tensor.dim(), (
            f"{name}: sharding spec {list(spec)} has {len(spec)} entries but the tensor is "
            f"{tensor.dim()}-D — a partition spec covers every dim."
        )

        placements: List[Placement] = []
        for mesh_dim, axis in enumerate(axis_names):
            tensor_dims = [dim for dim, entry in enumerate(spec) if entry == axis]
            if not tensor_dims:
                placements.append(Replicate())
                continue
            assert len(tensor_dims) == 1, (
                f"{name}: mesh axis '{axis}' appears on tensor dims {tensor_dims} in spec "
                f"{list(spec)}; one mesh axis can split only one tensor dim."
            )
            dim, num_shards = tensor_dims[0], self.mesh.size(mesh_dim)
            assert tensor.size(dim) % num_shards == 0, (
                f"{name}: dim {dim} (size {tensor.size(dim)}) is not divisible by mesh axis "
                f"'{axis}' ({num_shards}); tt collectives need an even split."
            )
            placements.append(Shard(dim))
        return placements


def _match(
    patterns: Sequence[Tuple[str, Sequence[Optional[str]]]], name: str
) -> Optional[Sequence[Optional[str]]]:
    """Partition spec of the first pattern matching `name`, or None."""
    return next((spec for pattern, spec in patterns if re.search(pattern, name)), None)


def _distribute_param(module: nn.Module, name: str, mesh, placements: List[Placement]) -> None:
    """Replace `module.<name>` in place with its distributed DTensor, keeping
    `requires_grad` (LoRA freezes the base weights and trains the adapters)."""
    param = getattr(module, name)
    distributed = distribute_tensor(param.detach(), mesh, placements)
    module.register_parameter(name, nn.Parameter(distributed, requires_grad=param.requires_grad))
