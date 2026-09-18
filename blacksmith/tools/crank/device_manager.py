# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device and parallelism setup for tt-crank.

Counterpart of `blacksmith.tools.device_manager.DeviceManager` (tt-xla). It keeps the
same public surface (`device`, `mesh`, `is_data_parallel()`, `is_tensor_parallel()`,
`is_fsdp()`, `shard_model`, `shard_tensor`, `prepare_batch`, `optimizer_step`) and
reads the *same* config fields (`mesh_shape`, `mesh_axis_names`, `input_sharding_dim`,
`model_sharding_patterns`), so a `train_crank.py` runs the experiment's existing YAML.

What differs underneath:

- tt-xla exposed one PJRT device per chip and used SPMD: an `xs.Mesh` plus
  `mark_sharding` annotations the XLA partitioner turned into collectives.
- tt-crank exposes a *single* logical `tt` device backed by a runtime `MeshDevice`.
  Parallelism is plain torch DTensor: `torch.tt.init_device_mesh` opens the mesh and
  builds a `DeviceMesh`, `distribute_tensor` / `distribute_module` place the shards,
  and collectives go through the `tt` c10d backend.
- Execution is eager: nothing here corresponds to `torch_xla.sync()` or
  `xm.optimizer_step`, and the lazy-graph workarounds (grad / AdamW-state
  pre-materialization, `capturable=True`) are not needed.

Not ported: FSDP (`SpmdFullyShardedDataParallel`). A mesh axis named "fsdp" is
accepted so the YAML validates, but its parameters are replicated, not sharded.
"""
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

from blacksmith.tools.templates.configs import TrainingConfig


def tt_compile_options(config) -> dict:
    """tt-crank options for `torch.compile(fn, backend="tt", options=...)`.

    tt-xla took these once, globally, via `torch_xla.set_custom_compile_options`
    (see the `if config.use_tt:` block at the bottom of a tt-xla `train.py`);
    tt-crank takes them per compiled callable. Same knobs, same defaults:
    fp32_dest_acc_en + HiFi4 for full-precision fine-tuning, the rest from the
    config. `experimental_weight_dtype` maps tt-xla's "bfp_bf8" / "bfp_bf4" to
    tt-crank's `BfpDtype`; "bf16" (the tt-xla default) means no override.
    """
    from tt_crank.torch._compile import BfpDtype, CompileOption, MathFidelity

    options = {
        CompileOption.FP32_DEST_ACC_EN: getattr(config, "fp32_dest_acc_en", True),
        CompileOption.MATH_FIDELITY: getattr(MathFidelity, getattr(config, "math_fidelity", "HiFi4")),
        CompileOption.OPT_LEVEL: getattr(config, "optimization_level", 0),
        CompileOption.ENABLE_CONST_EVAL: getattr(config, "enable_const_eval", True),
        CompileOption.ENABLE_TRACE: getattr(config, "enable_trace", False),
    }
    weight_dtype = getattr(config, "experimental_weight_dtype", None)
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
        """Bring up the "tt" c10d process group that DTensor collectives use.

        Single process, one rank: the collectives run inside the device mesh, so
        there are no peer processes to rendezvous with and a `FakeStore` stands in
        for the TCP store. `world_size` is the mesh size, not the chip count.
        """
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
        """tt-xla partition spec -> DTensor placements.

        `sharding_spec` has one entry per *tensor* dim naming the mesh axis that dim
        is sharded along (None = replicated), exactly what `xs.mark_sharding` took.
        DTensor placements are per *mesh* dim instead, so this transposes the two.
        """
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
        """Place `tensor` on the mesh given a tt-xla style partition spec.

        Returns a DTensor: unlike `xs.mark_sharding`, which annotated in place, the
        caller has to use the return value.
        """
        if self.mesh is None:
            return tensor
        return distribute_tensor(tensor, self.mesh, self._placements(sharding_spec))

    def shard_model(self, model: nn.Module) -> nn.Module:
        """Turn the model's parameters into DTensors over the mesh, in place.

        `model_sharding_patterns` (regex on the module name -> partition spec) shard
        `module.weight` exactly as the tt-xla version did, `param_sharding_patterns`
        likewise by parameter name. Every other parameter becomes an explicitly
        replicated DTensor, which is what makes both TP and pure DP work: no op ever
        sees a DTensor and a plain tensor together.

        Single chip: no mesh, nothing to do. Idempotent: the tt-xla scripts call
        this every step, and a DTensor parameter stays a DTensor, so after the first
        call this is a no-op.
        """
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
        """Step the optimizer, optionally zeroing grads afterwards.

        The data-parallel gradient all-reduce is implicit: under DP the loss is a
        `Partial` DTensor, autograd produces `Partial` grads and the optimizer's
        first read redistributes them. No `xm.optimizer_step`, no barrier. Grads
        are released rather than re-zeroed in place; eager execution has no graph
        signature to keep stable.
        """
        optimizer.step()
        if zero_grad:
            optimizer.zero_grad(set_to_none=True)

    def compile_options(self) -> dict:
        """See `tt_compile_options`."""
        return tt_compile_options(self.config)

    def replication_context(self):
        """Context in which plain tensors are treated as replicated DTensors.

        HF builds a few tensors internally (causal-mask helpers, position ids)
        that never pass through `prepare_batch`. Without this they stay plain
        tensors and mixing them with DTensor parameters raises. No-op on a
        single chip.
        """
        if self.mesh is None:
            return contextlib.nullcontext()

        from torch.distributed.tensor.experimental import implicit_replication

        return implicit_replication()
