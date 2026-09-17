# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device and parallelism setup for tt-crank.

This is the tt-crank counterpart of the tt-xla DeviceManager and keeps its
public surface (`device`, `mesh`, `is_data_parallel()`, `is_tensor_parallel()`,
`is_fsdp()`, `shard_model`, `shard_tensor`, `prepare_batch`, `optimizer_step`)
so experiment scripts port over with import changes only. The two differ in
kind underneath, not just in API:

- tt-xla exposed one PJRT device per chip and used SPMD: an `xs.Mesh` plus
  `mark_sharding` annotations that the XLA partitioner turned into collectives.
- tt-crank exposes a *single* logical `tt` device backed by a runtime
  `MeshDevice`. Parallelism is plain torch DTensor: `torch.tt.init_device_mesh`
  opens the mesh and builds a `DeviceMesh`, then `distribute_tensor` /
  `distribute_module` place shards. Collectives come from the `tt` c10d backend.

There is also no lazy-execution fence: tt-crank is eager, so nothing here
corresponds to `torch_xla.sync()` / `xm.optimizer_step`.
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
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)

from blacksmith.tools.templates.configs import TrainingConfig

# Megatron tensor parallelism, matched against each leaf module's fully
# qualified name as `distribute_module` reports it.
#
# Column-parallel linears shard the output (feature) dim; row-parallel linears
# shard the contraction (input) dim. LoRA splits each target into three
# `nn.Linear`s -- `base_layer`, `lora_A` (r x in) and `lora_B` (out x r) -- and
# they do NOT all follow the parent's rule:
#
#   column-parallel (out sharded): base_layer and lora_B shard dim 0;
#       lora_A stays replicated (its output is the rank dim, not the feature dim).
#   row-parallel (in sharded): base_layer and lora_A shard dim 1;
#       lora_B stays replicated (its input is the rank dim).
#
# An unadapted target is a bare `nn.Linear`, hence the optional suffix group.
# `embed_tokens` / `lm_head` are deliberately absent: Llama ties them, so
# vocab-sharding lm_head would also shard the embedding lookup and corrupt it.
_COLUMN_PARALLEL = (
    re.compile(r"\.(q_proj|k_proj|v_proj|gate_proj|up_proj)(\.base_layer|\.lora_B\.[^.]+)?$"),
    Shard(0),
)
_ROW_PARALLEL = (
    re.compile(r"\.(o_proj|down_proj)(\.base_layer|\.lora_A\.[^.]+)?$"),
    Shard(1),
)

_TENSOR_PARALLEL_RULES = (_COLUMN_PARALLEL, _ROW_PARALLEL)


def tt_compile_options(config) -> dict:
    """tt-crank compile options for `torch.compile(backend="tt", options=...)`.

    tt-xla set these once, globally, via `set_custom_compile_options`; tt-crank
    takes them per compiled callable. Reads the fields with defaults so it works
    for both the flat `TrainingConfig` and the nested `TrainerConfig`.
    """
    from tt_crank.torch._compile import BfpDtype, CompileOption, MathFidelity

    options = {
        CompileOption.OPT_LEVEL: getattr(config, "optimization_level", 0),
        CompileOption.MATH_FIDELITY: getattr(MathFidelity, getattr(config, "math_fidelity", "HiFi4")),
        CompileOption.FP32_DEST_ACC_EN: getattr(config, "fp32_dest_acc_en", True),
        CompileOption.ENABLE_CONST_EVAL: getattr(config, "enable_const_eval", True),
        CompileOption.ENABLE_TRACE: getattr(config, "enable_trace", False),
    }
    weight_dtype = getattr(config, "experimental_weight_dtype", None)
    if weight_dtype:
        options[CompileOption.EXPERIMENTAL_WEIGHT_DTYPE] = getattr(BfpDtype, weight_dtype)
    return options


class DeviceManager:
    """Owns the `tt` device and, for multichip runs, the DTensor mesh."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.mesh: Optional[DeviceMesh] = None

        if not config.use_tt:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return

        # Importing the package registers the "tt" PrivateUse1 backend, the
        # "tt" dynamo backend and the "tt" c10d backend. Nothing else is needed;
        # in particular there is no PJRT_DEVICE / XLA_* environment to set.
        import tt_crank.torch  # noqa: F401

        self.device = torch.device("tt")

        if getattr(config, "mesh", None) is not None:
            if "fsdp" in config.mesh.axis_names:
                warnings.warn(
                    "mesh axis 'fsdp' requested but FSDP is not ported to tt-crank (see DeviceManager.is_fsdp); "
                    "parameters along that axis will be replicated.",
                    stacklevel=2,
                )
            self._init_mesh()

    def _init_mesh(self) -> None:
        mesh_cfg = self.config.mesh
        available = torch.tt.num_chips()
        requested = math.prod(mesh_cfg.shape)
        assert requested <= available, f"mesh {mesh_cfg.shape} needs {requested} chips, only {available} available"

        self._init_process_group(requested)

        # init_device_mesh opens the runtime MeshDevice at this shape *and*
        # builds the torch DeviceMesh; the two must not be set up separately.
        self.mesh = torch.tt.init_device_mesh(tuple(mesh_cfg.shape), mesh_dim_names=tuple(mesh_cfg.axis_names))

    @staticmethod
    def _init_process_group(world_size: int) -> None:
        """Bring up the "tt" c10d process group that DTensor collectives use.

        Single-process, one rank per chip: this is not distributed training in
        the usual sense -- the collectives run inside the device mesh, so there
        are no peer processes to rendezvous with and a `FakeStore` stands in for
        the TCP store. `world_size` is the mesh size rather than the machine's
        chip count, so a run that uses part of a larger machine still gets a
        world that matches the mesh it opened.
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

    # Methods rather than properties: that is how the tt-xla DeviceManager
    # exposed them and how the experiment scripts call them.
    def is_data_parallel(self) -> bool:
        cfg = self.config.mesh
        return self.mesh is not None and cfg.data_axis is not None and cfg.axis_size(cfg.data_axis) > 1

    def is_tensor_parallel(self) -> bool:
        cfg = self.config.mesh
        return self.mesh is not None and cfg.tensor_axis is not None and cfg.axis_size(cfg.tensor_axis) > 1

    def is_fsdp(self) -> bool:
        """FSDP is not ported to tt-crank yet; always False.

        tt-xla used `SpmdFullyShardedDataParallel` (torch_xla.experimental) keyed on a mesh axis
        literally named "fsdp", with a `shard_output` hook on the logits. The DTensor equivalent is
        `torch.distributed.fsdp.fully_shard(model, mesh=self.mesh["fsdp"])`, but that relies on
        c10d reduce_scatter / all_gather through the `tt` process-group backend, which has not been
        exercised here. Until it is, a mesh whose axis_names contain "fsdp" is accepted (so old YAMLs
        validate) but its parameters are replicated, not sharded -- memory use matches DP, not FSDP.
        `__init__` warns when that happens. tt-blacksmith issue to track: FSDP on tt-crank.
        """
        return False

    def _placements(self, axis: str, shard: Shard) -> list:
        """Shard along `axis`, replicate every other mesh axis."""
        idx = self.config.mesh.axis_index(axis)
        return [shard if i == idx else Replicate() for i in range(len(self.config.mesh.shape))]

    def shard_tensor(self, tensor: torch.Tensor, sharding_spec: Sequence[Optional[str]]) -> torch.Tensor:
        """Place `tensor` on the mesh given a tt-xla style partition spec.

        `sharding_spec` has one entry per tensor dim: a mesh axis name to shard
        that dim along, or None to replicate it -- the same tuple `xs.mark_sharding`
        took. Returns a DTensor (tt-xla annotated in place; DTensor is a new object,
        so callers must use the return value).
        """
        if self.mesh is None:
            return tensor
        placements = [Replicate() for _ in self.config.mesh.shape]
        for dim, axis in enumerate(sharding_spec):
            if axis is None:
                continue
            placements[self.config.mesh.axis_index(axis)] = Shard(dim)
        return distribute_tensor(tensor, self.mesh, placements)

    def shard_model(self, model: nn.Module) -> nn.Module:
        """Turn the model's parameters into DTensors over the mesh, in place.

        `distribute_module` replicates every parameter it is not told to shard,
        which is what makes the mixed case work: under tensor parallelism the
        matched projections are column/row sharded and everything else (norms,
        embeddings, the LoRA A matrices) becomes an explicitly replicated
        DTensor, so no op ever sees a DTensor and a plain tensor together.

        Under pure data parallelism there is nothing to shard, so this is a
        plain replicate -- the batch is what gets sharded, in `prepare_batch`.

        Single chip: no mesh, nothing to do.

        Idempotent: a DTensor parameter stays a DTensor, so unlike the tt-xla
        SPMD version calling this every step (as the old scripts do) is a no-op
        after the first call.
        """
        if self.mesh is None:
            return model
        if self._is_distributed(model):
            return model

        # Every parameter is placed by our partition_fn (sharded if a TP rule matches, replicated
        # otherwise). torch's own replicate pass in `distribute_module` only handles parameters the
        # partition_fn left alone, and it rebuilds them as bare `nn.Parameter(...)`, i.e. with
        # requires_grad=True -- under LoRA that would turn the frozen base model trainable. Placing
        # everything ourselves keeps the flag and leaves that pass nothing to do (buffers excepted,
        # which it replicates and which carry no grad flag).
        return distribute_module(model, self.mesh, partition_fn=self._partition_fn())

    # tt-crank name for the same operation.
    distribute_model = shard_model

    @staticmethod
    def _is_distributed(model: nn.Module) -> bool:
        first = next(model.parameters(), None)
        return isinstance(first, DTensor)

    def _partition_fn(self):
        """Place each module's direct parameters on the mesh, preserving requires_grad.

        Megatron column/row sharding for each decoder layer's projections when tensor
        parallelism is on; everything else (norms, embeddings, LoRA A/B that the rules
        leave out) is replicated. `embed_tokens` / `lm_head` are deliberately not
        matched: Llama ties them, so vocab-sharding lm_head would also shard the
        embedding lookup.
        """
        tensor_axis = self.config.mesh.tensor_axis if self.is_tensor_parallel() else None
        replicate = [Replicate() for _ in self.config.mesh.shape]

        def partition_fn(name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
            # A PEFT adapter wrapper holds the real linear in `base_layer`; its own `.weight` is
            # a property forwarding to that child, so let the child match the TP rule instead.
            shard = None
            if tensor_axis is not None and isinstance(module, nn.Linear) and not hasattr(module, "base_layer"):
                shard = next((s for pattern, s in _TENSOR_PARALLEL_RULES if pattern.search(name)), None)

            for param_name, param in list(module._parameters.items()):
                if param is None or isinstance(param, DTensor):
                    continue
                # Column-parallel: the bias follows the sharded output features. Row-parallel: the
                # output is a Partial sum, so the bias stays replicated and is added after the reduce.
                if shard is not None and (param_name == "weight" or (param_name == "bias" and shard == Shard(0))):
                    placements = self._placements(tensor_axis, shard)
                else:
                    placements = replicate
                module.register_parameter(
                    param_name,
                    nn.Parameter(distribute_tensor(param, device_mesh, placements), requires_grad=param.requires_grad),
                )

        return partition_fn

    def shard_optimizer(self, optimizer: torch.optim.Optimizer):
        raise NotImplementedError("Optimizer sharding is not implemented yet.")

    def prepare_batch(
        self,
        batch: Dict[str, torch.Tensor],
        skip_keys: tuple[str, ...] = (),
    ) -> Dict[str, torch.Tensor]:
        """Move the batch to device, sharding the batch dim under data parallelism.

        ``skip_keys`` stay on the host.
        """
        batch = {k: v if k in skip_keys else v.to(self.device) for k, v in batch.items()}

        if not self.is_data_parallel():
            return batch

        placements = self._placements(self.config.mesh.data_axis, Shard(0))
        return {
            k: v if k in skip_keys or v.dim() == 0 else distribute_tensor(v, self.mesh, placements)
            for k, v in batch.items()
        }

    def optimizer_step(self, optimizer: torch.optim.Optimizer, zero_grad: bool = False) -> None:
        """Step the optimizer, optionally zeroing grads afterwards.

        Gradient all-reduce across the data-parallel axis is implicit: under DP
        the loss is a `Partial` DTensor, so autograd produces `Partial` grads and
        the optimizer's first read redistributes them. No `xm.optimizer_step`
        equivalent and no barrier -- tt-crank executes eagerly.

        ``zero_grad`` keeps the tt-xla signature. There it re-zeroed in place so
        the grads stayed tensors for graph-signature stability; eager execution
        has no such constraint, so grads are simply released.
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
        tensors and mixing them with DTensor parameters raises. A no-op on a
        single chip.
        """
        if self.mesh is None:
            return contextlib.nullcontext()

        from torch.distributed.tensor.experimental import implicit_replication

        return implicit_replication()
