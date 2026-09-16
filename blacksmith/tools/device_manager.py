# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device and parallelism setup for tt-crank.

This is the tt-crank counterpart of the tt-xla DeviceManager. The two differ
in kind, not just in API:

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
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, Replicate, Shard, distribute_module, distribute_tensor

from blacksmith.tools.configs import TrainingConfig

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

        if config.mesh is not None:
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

    @property
    def is_data_parallel(self) -> bool:
        cfg = self.config.mesh
        return self.mesh is not None and cfg.data_axis is not None and cfg.axis_size(cfg.data_axis) > 1

    @property
    def is_tensor_parallel(self) -> bool:
        cfg = self.config.mesh
        return self.mesh is not None and cfg.tensor_axis is not None and cfg.axis_size(cfg.tensor_axis) > 1

    def _placements(self, axis: str, shard: Shard) -> list:
        """Shard along `axis`, replicate every other mesh axis."""
        idx = self.config.mesh.axis_index(axis)
        return [shard if i == idx else Replicate() for i in range(len(self.config.mesh.shape))]

    def distribute_model(self, model: nn.Module) -> nn.Module:
        """Turn the model's parameters into DTensors over the mesh, in place.

        `distribute_module` replicates every parameter it is not told to shard,
        which is what makes the mixed case work: under tensor parallelism the
        matched projections are column/row sharded and everything else (norms,
        embeddings, the LoRA A matrices) becomes an explicitly replicated
        DTensor, so no op ever sees a DTensor and a plain tensor together.

        Under pure data parallelism there is nothing to shard, so this is a
        plain replicate -- the batch is what gets sharded, in `prepare_batch`.

        Single chip: no mesh, nothing to do.

        Call once, before training. Unlike the tt-xla SPMD version there is no
        per-step re-annotation: a DTensor parameter stays a DTensor.
        """
        if self.mesh is None:
            return model

        partition_fn = self._tensor_parallel_partition_fn() if self.is_tensor_parallel else None
        return distribute_module(model, self.mesh, partition_fn=partition_fn)

    def _tensor_parallel_partition_fn(self):
        """Megatron column/row sharding for each decoder layer's projections.

        `embed_tokens` / `lm_head` are deliberately not matched: Llama ties them,
        so vocab-sharding lm_head would also shard the embedding lookup.
        """
        axis = self.config.mesh.tensor_axis

        def partition_fn(name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
            if not isinstance(module, nn.Linear):
                return
            # A PEFT adapter wrapper holds the real linear in `base_layer`; its
            # own `.weight` is a property forwarding to that child, so sharding
            # it here would double-apply. Let the children match instead.
            if hasattr(module, "base_layer"):
                return
            shard = next((s for pattern, s in _TENSOR_PARALLEL_RULES if pattern.search(name)), None)
            if shard is None:
                return
            weight = module.weight
            module.register_parameter(
                "weight",
                nn.Parameter(
                    distribute_tensor(weight, device_mesh, self._placements(axis, shard)),
                    requires_grad=weight.requires_grad,
                ),
            )

        return partition_fn

    def prepare_batch(
        self,
        batch: Dict[str, torch.Tensor],
        skip_keys: tuple[str, ...] = (),
    ) -> Dict[str, torch.Tensor]:
        """Move the batch to device, sharding the batch dim under data parallelism."""
        batch = {k: v if k in skip_keys else v.to(self.device) for k, v in batch.items()}

        if not self.is_data_parallel:
            return batch

        placements = self._placements(self.config.mesh.data_axis, Shard(0))
        return {
            k: v if k in skip_keys or v.dim() == 0 else distribute_tensor(v, self.mesh, placements)
            for k, v in batch.items()
        }

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """Step the optimizer.

        Gradient all-reduce across the data-parallel axis is implicit: under DP
        the loss is a `Partial` DTensor, so autograd produces `Partial` grads and
        the optimizer's first read redistributes them. No `xm.optimizer_step`
        equivalent and no barrier -- tt-crank executes eagerly.
        """
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def compile_options(self) -> dict:
        """tt-crank compile options for `torch.compile(backend="tt", options=...)`.

        tt-xla set these once, globally, via `set_custom_compile_options`;
        tt-crank takes them per compiled callable.
        """
        from tt_crank.torch._compile import BfpDtype, CompileOption, MathFidelity

        options = {
            CompileOption.OPT_LEVEL: self.config.optimization_level,
            CompileOption.MATH_FIDELITY: getattr(MathFidelity, self.config.math_fidelity),
            CompileOption.FP32_DEST_ACC_EN: self.config.fp32_dest_acc_en,
            CompileOption.ENABLE_CONST_EVAL: self.config.enable_const_eval,
            CompileOption.ENABLE_TRACE: self.config.enable_trace,
        }
        if self.config.experimental_weight_dtype:
            options[CompileOption.EXPERIMENTAL_WEIGHT_DTYPE] = getattr(BfpDtype, self.config.experimental_weight_dtype)
        return options

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
