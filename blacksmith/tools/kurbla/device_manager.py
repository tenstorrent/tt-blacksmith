# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

import tt_kurbla.torch  # noqa: F401  — registers the "tt" device, c10d backend, dynamo backend, torch.tt
from tt_kurbla.torch._compile import CompileOption, MathFidelity

DEFAULT_COMPILE_OPTIONS = {
    CompileOption.FP32_DEST_ACC_EN: True,
    CompileOption.MATH_FIDELITY: MathFidelity.HiFi4,
    CompileOption.EXPERIMENTAL_ENABLE_DRAM_SPACE_SAVING_OPTIMIZATION: True,
}

_dtensor_pad_patched = False


def _patch_dtensor_pad() -> None:
    """Make `F.pad` DTensor-aware: its `constant_pad_nd` strategy is a 1-D-mesh stub."""
    global _dtensor_pad_patched
    if _dtensor_pad_patched:
        return
    _dtensor_pad_patched = True

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
        input: torch.Tensor, pad: List[int], mode: str = "constant", value: Optional[float] = None
    ) -> torch.Tensor:
        if not torch.jit.is_scripting():
            return _pad_maybe_dtensor(input, pad, mode, value)
        return _orig_pad(input, pad, mode=mode, value=value)

    F.pad = _pad_dtensor_safe


class DeviceManager:
    """tt-kurbla device, mesh and DTensor sharding. See the module docstring for the config
    contract.

    Args:
        config: anything carrying the fields above; only `use_tt` is required.
        compile_options: replaces `DEFAULT_COMPILE_OPTIONS` wholesale when given.
        model_rewrites: called by `prepare_model` on a freshly built model, for backend gaps
            that must be patched per module instance rather than per class.
        verbose: log one line per parameter in `shard_model`. Off by default — a large model
            produces tens of thousands of lines; the summary is always logged.
    """

    def __init__(
        self,
        config: Any,
        *,
        compile_options: Optional[dict] = None,
        model_rewrites: Optional[Callable[[nn.Module], Any]] = None,
        verbose: bool = False,
    ):
        self.config = config
        self.mesh: Optional[DeviceMesh] = None
        self._compile_cache: dict = {}
        self._model_rewrites = model_rewrites
        self.verbose = verbose

        # The whole config contract, resolved once. Everything below reads these, not
        # `self.config`, so the manager works with any config object and callers can
        # override a field programmatically before use.
        self.use_tt: bool = getattr(config, "use_tt", True)
        self.mesh_shape: Optional[Sequence[int]] = getattr(config, "mesh_shape", None)
        self.mesh_axis_names: Optional[Sequence[str]] = getattr(config, "mesh_axis_names", None)
        self.input_sharding_dim: Optional[str] = getattr(config, "input_sharding_dim", None)
        self.model_sharding_patterns = getattr(config, "model_sharding_patterns", None) or []
        self.param_sharding_patterns = getattr(config, "param_sharding_patterns", None) or []
        self.compile_optimizer: bool = getattr(config, "compile_optimizer", False)

        base_options = DEFAULT_COMPILE_OPTIONS if compile_options is None else compile_options
        self.compile_options: dict = dict(base_options) | {
            CompileOption.OPT_LEVEL: getattr(config, "optimization_level", 0),
        }

        self._setup()

    def _setup(self) -> None:
        if not self.use_tt:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return

        self.device = torch.device("tt")
        self.mesh = self._create_mesh()

    def _create_mesh(self) -> Optional[DeviceMesh]:
        """Open the mesh described by `mesh_shape`/`mesh_axis_names`, or None for one chip. """
        if not self.mesh_shape:
            return None

        assert self.mesh_axis_names is not None, "`mesh_axis_names` must be provided alongside `mesh_shape`."
        assert len(self.mesh_shape) == len(self.mesh_axis_names), (
            f"`mesh_shape` {list(self.mesh_shape)} and `mesh_axis_names` "
            f"{list(self.mesh_axis_names)} must have the same length."
        )
        assert (self.input_sharding_dim is None) or (self.input_sharding_dim in self.mesh_axis_names), (
            f"`input_sharding_dim` {self.input_sharding_dim!r} is not one of the mesh axes "
            f"{list(self.mesh_axis_names)}."
        )

        mesh_size = math.prod(self.mesh_shape)
        num_chips = torch.tt.num_chips()
        assert mesh_size <= num_chips, f"Mesh {tuple(self.mesh_shape)} needs {mesh_size} chips, found {num_chips}."

        if not dist.is_initialized():
            dist.init_process_group(backend="tt", rank=0, world_size=mesh_size, store=dist.HashStore())

        _patch_dtensor_pad()

        try:
            return torch.tt.init_device_mesh(
                tuple(self.mesh_shape), mesh_dim_names=tuple(self.mesh_axis_names)
            )
        except RuntimeError as e:
            # The runtime raises from deep in the fabric stack with a long backtrace and no
            # mention of the env var that governs it. Two distinct failures land here:
            #   * the topology mapper cannot fit this shape onto the discovered chips;
            #   * the descriptor's topology is weaker than the fabric config wants (a
            #     line/MESH descriptor cannot satisfy a TORUS request — fabric config can
            #     restrict connectivity, not create it).
            # Both are fixed by pointing at a descriptor that matches this machine *and*
            # the topology the runtime asks for.
            import os

            mgd = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
            raise RuntimeError(
                f"Opening a {tuple(self.mesh_shape)} mesh ({mesh_size} of {num_chips} visible chips) "
                f"failed in the fabric layer. TT_MESH_GRAPH_DESC_PATH is "
                f"{mgd or 'unset, so the control plane auto-discovers the mesh graph — which does not '
                          'always match the physical topology'}. "
                f"Check the original error above: if it names a topology mismatch, the descriptor "
                f"declares less connectivity (e.g. line/MESH) than the fabric config requests "
                f"(e.g. TORUS_X) and a different descriptor is needed. Or set the mesh shape to null "
                f"to run on one chip."
            ) from e

    def is_data_parallel(self) -> bool:
        """True when a mesh axis is configured to split the batch and is wider than one chip."""
        if self.input_sharding_dim is None or self.mesh is None:
            return False
        return self.mesh.size(self.mesh.mesh_dim_names.index(self.input_sharding_dim)) > 1

    def is_tensor_parallel(self) -> bool:
        """True when a mesh is open and at least one sharding pattern is configured. """
        return bool(self.model_sharding_patterns or self.param_sharding_patterns) and self.mesh is not None

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
        """Apply per-instance graph rewrites to a freshly constructed model.

        Some backend gaps have to be worked around by retyping or replacing individual
        module instances, which a class-level patch cannot do. The rewrite function is
        injected (`model_rewrites=`) so this stays model-agnostic; without one this is a
        no-op. Call it before `to_device`/`shard_model`, so the device only ever sees
        modules the backend can lower.
        """
        if self._model_rewrites is None:
            return model
        result = self._model_rewrites(model)
        # Tolerate a rewrite that mutates in place and returns None, or one that returns
        # a report (e.g. a dict of what it changed) rather than the model.
        return result if isinstance(result, nn.Module) else model

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Full, mesh-wide plain tensor from a (possibly distributed) one.

        The inverse of `to_device`/`prepare_batch`, for values leaving the device:
        `full_tensor()` fires whatever collective the placement needs (all-gather for
        `Shard`, all-reduce for `Partial`). A no-op on plain tensors.
        """
        return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor

    def shard_tensor(self, tensor: torch.Tensor, sharding_spec: Sequence[Optional[str]]) -> DTensor:
        """Distribute `tensor` per a partition spec (one mesh axis name, or None, per tensor dim)."""
        return distribute_tensor(tensor, self.mesh, self._placements(sharding_spec, tensor))

    def shard_model(self, model: nn.Module) -> nn.Module:
        """Distribute every parameter and buffer of `model` over the mesh.

        Params named by `model_sharding_patterns` (matched on module name, applied to
        `module.weight`) or `param_sharding_patterns` (matched on the full parameter
        name) get that pattern's placements; everything else is replicated. DTensor
        conversion is all-or-nothing — a single plain parameter left behind makes the
        first op touching it raise on mixed operands — so unmatched tensors are
        replicated rather than skipped.

        Logs a sharded/replicated tally: a pattern list that matches nothing leaves the
        model fully replicated and still produces correct results, so a run that believes
        it is tensor-parallel needs a way to see that it is not.
        """
        if self.mesh is None:
            return model

        module_patterns = self.model_sharding_patterns
        param_patterns = self.param_sharding_patterns
        full_replicate = [Replicate()] * self.mesh.ndim
        counts = {"sharded": 0, "replicated": 0}

        def partition_fn(name: str, module: nn.Module, device_mesh) -> None:
            # Module patterns address `module.weight`; a param pattern can name any
            # parameter. Module patterns win on `weight`, so a param pattern cannot
            # silently override a module-level decision.
            weight_spec = _match(module_patterns, name)

            for param_name, param in list(module.named_parameters(recurse=False)):
                if isinstance(param, DTensor):  # already distributed (shard_model re-run)
                    continue
                qualified_name = f"{name}.{param_name}" if name else param_name
                spec = weight_spec if param_name == "weight" else None
                if spec is None:
                    spec = _match(param_patterns, qualified_name)
                placements = full_replicate if spec is None else self._placements(spec, param, qualified_name)
                counts["sharded" if any(p.is_shard() for p in placements) else "replicated"] += 1
                # Replicate here too rather than leaving it to `distribute_module`: its
                # fallback rebuilds the Parameter with the default requires_grad=True,
                # which would un-freeze frozen (e.g. LoRA base) weights.
                if self.verbose:
                    print(f"[shard_model] {qualified_name}: {tuple(param.shape)} -> {placements}")
                _distribute_param(module, param_name, device_mesh, placements)

        # Buffers (unmatched by construction — the patterns name parameters) are
        # replicated by `distribute_module` itself.
        sharded = distribute_module(model.to(self.device), self.mesh, partition_fn=partition_fn)
        print(
            f"[shard_model] {type(model).__name__}: {counts['sharded']} sharded / "
            f"{counts['replicated']} replicated over mesh {tuple(self.mesh_shape)}",
            flush=True,
        )
        return sharded

    def prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move a batch to device, splitting dim 0 over `input_sharding_dim` when set.

        Note the batch dim must then be divisible by that axis's width, so the loader has
        to yield the *global* batch (per-device batch x data-parallel width).
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.mesh is None:
            return batch

        shard_batch_dim = self.is_data_parallel()
        prepared = {}
        for name, tensor in batch.items():
            if shard_batch_dim and tensor.dim() > 0:
                spec = [self.input_sharding_dim] + [None] * (tensor.dim() - 1)
                prepared[name] = self.shard_tensor(tensor, spec)
            else:
                prepared[name] = self._replicate(tensor)
        return prepared

    def compile(self, module: nn.Module):
        """`torch.compile(backend="tt")`, cached on id(module).

        Callers must keep the returned wrapper alive across calls: building a fresh
        `OptimizedModule` around an already-compiled module on every step defeats the
        point, which is what the cache prevents.
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
        """Step the optimizer, as one compiled graph when `compile_optimizer` is set.

        No explicit all-reduce: with DTensor parameters the gradients are DTensors too,
        and a data-parallel gradient comes out `Partial` over the batch axis. The step
        combines it with the replicated parameter, so DTensor emits the reduction itself.

        Run eagerly, the step is one program submit per parameter per op. Dynamo traces
        the whole of AdamW's single-tensor path without a graph break, so compiling
        `optimizer.step` turns those into one graph. Cached on id(optimizer) like
        `compile`, because a fresh wrapper each call would defeat the point.
        """
        if not self.compile_optimizer:
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

        Kept so training loops can call it unconditionally.
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

        A partition spec has one entry *per tensor dim*, naming the mesh axis that splits
        it (or None to leave it whole). DTensor is indexed the other way round: one
        placement *per mesh dim*, naming the tensor dim that axis splits. So
        `["model", "batch"]` on a 2-D weight over a ("batch", "model") mesh becomes
        `[Shard(1), Shard(0)]` — mesh axis "batch" splits tensor dim 1, "model" splits
        dim 0.
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
            # A size-1 axis divides everything, so without this the spec would produce a
            # Shard() over one chip: a no-op that reads as sharding and still gives correct
            # results, hiding a mesh or spec that is not what was intended.
            assert num_shards > 1, (
                f"{name}: sharding spec {list(spec)} splits dim {dim} over mesh axis '{axis}', "
                f"which has size 1; that shards nothing. Drop the axis from the spec, or widen it."
            )
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
    `requires_grad` (frozen base weights must stay frozen)."""
    param = getattr(module, name)
    distributed = distribute_tensor(param.detach(), mesh, placements)
    module.register_parameter(name, nn.Parameter(distributed, requires_grad=param.requires_grad))
