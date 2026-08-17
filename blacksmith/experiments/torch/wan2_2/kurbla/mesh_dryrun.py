# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU dry run of a config's DTensor sharding — the rung below `mesh_ladder.py`.

`mesh_ladder.py` needs chips and minutes. This needs neither: it runs each component's
forward under the config's real mesh and sharding patterns, on CPU, in seconds, and
reports where DTensor rejects a placement.

    P=blacksmith.experiments.torch.wan2_2.kurbla.mesh_dryrun
    CFG=blacksmith/experiments/torch/wan2_2/kurbla/lora/multi_chip/wan2_2_ti2v_5b_diffusiondb.yaml

    python -m $P --config $CFG                      # every stage
    python -m $P --config $CFG --stage dit          # one stage
    python -m $P --config $CFG --no-overrides       # what breaks without model_overrides
    python -m $P --config $CFG --json out.json      # machine-readable result

Three things make it fast. torch's `fake` process-group backend gives a full N-rank world
inside one process, so a 32-chip mesh needs no chips and collectives become no-ops.
Weights are built on `meta` from the configs in `bringup.py`, so nothing is downloaded,
allocated or initialised. And the forward runs eager, so a failure points at model code
instead of a hundred frames of dynamo.

**What this proves:** DTensor sharding propagation — every "got mixed torch.Tensor and
DTensor", "cannot be performed without redistribution", "in-place operations that require
placement changes", and every divisibility assert in a sharding spec.

**What it does not prove:** values (meta tensors carry none), and anything about the tt
backend — a missing lowering or an unlowerable collective still only shows up on hardware.
Use `bringup.py` for pcc and `mesh_ladder.py` for the real thing.

The sharding here mirrors `WanDeviceManager.shard_model` / `_placements` and
`_patch_dtensor_conv` rather than importing them, so this tool stays runnable without
`tt_kurbla` installed. If those change, change `_placements_for` and `_patch_conv_support`
with them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.distributed as dist
import yaml
from torch.distributed.tensor import DeviceMesh, DTensor, Placement, Replicate, Shard, distribute_tensor

from blacksmith.experiments.torch.wan2_2.kurbla.bringup import (
    WAN22_TI2V_5B_DIT_CONFIG,
    WAN22_TI2V_5B_VAE_CONFIG,
)

STAGES = ("dit", "umt5", "vae-encode", "vae-decode")
VAE_SPATIAL_STRIDE = 16
VAE_TEMPORAL_STRIDE = 4


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Experiment YAML; its mesh_shape and sharding patterns are used.")
    p.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    p.add_argument("--layers", type=int, default=None, help="DiT blocks (default: the config's full 30).")
    p.add_argument("--h", type=int, default=None, help="Pixel height (default: train_h).")
    p.add_argument("--w", type=int, default=None, help="Pixel width (default: train_w).")
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--text-len", type=int, default=512)
    p.add_argument("--backward", action="store_true",
                   help="Also run a backward pass (dit stage only): catches sharding gaps that only "
                        "the autograd graph hits, e.g. aten.linear_backward.")
    p.add_argument("--no-overrides", action="store_true", help="Skip apply_generality_overrides().")
    p.add_argument("--full-traceback", action="store_true")
    p.add_argument("--json", default=None, help="Write the per-stage result to this path.")
    return p.parse_args(argv)


# --- sharding, mirroring WanDeviceManager ------------------------------------------


def _match(patterns, name: str):
    return next((spec for pattern, spec in patterns or [] if re.search(pattern, name)), None)


def _placements_for(spec: Sequence[Optional[str]], tensor: torch.Tensor, mesh: DeviceMesh, name: str):
    """A YAML spec has one entry per *tensor* dim naming the mesh axis that splits it;
    DTensor wants one placement per *mesh* dim naming the tensor dim it splits."""
    axis_names = mesh.mesh_dim_names
    assert len(spec) == tensor.dim(), f"{name}: spec {list(spec)} has {len(spec)} entries for a {tensor.dim()}-D tensor"
    out: list[Placement] = []
    for mesh_dim, axis in enumerate(axis_names):
        dims = [d for d, entry in enumerate(spec) if entry == axis]
        if not dims:
            out.append(Replicate())
            continue
        assert len(dims) == 1, f"{name}: mesh axis '{axis}' splits tensor dims {dims}"
        size, shards = tensor.size(dims[0]), mesh.size(mesh_dim)
        assert size % shards == 0, f"{name}: dim {dims[0]} (size {size}) not divisible by '{axis}' ({shards})"
        out.append(Shard(dims[0]))
    return out


def shard_model(model: nn.Module, mesh: DeviceMesh, cfg: dict) -> nn.Module:
    module_patterns = cfg.get("model_sharding_patterns")
    param_patterns = cfg.get("param_sharding_patterns")
    full_replicate = [Replicate()] * mesh.ndim

    for name, module in model.named_modules():
        weight_spec = _match(module_patterns, name)
        for param_name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param, DTensor):
                continue
            qualified = f"{name}.{param_name}" if name else param_name
            spec = weight_spec if param_name == "weight" else None
            if spec is None:
                spec = _match(param_patterns, qualified)
            placements = full_replicate if spec is None else _placements_for(spec, param, mesh, qualified)
            module.register_parameter(
                param_name,
                nn.Parameter(distribute_tensor(param.detach(), mesh, placements), requires_grad=False),
            )
        for buf_name, buf in list(module.named_buffers(recurse=False)):
            # distribute_module replicates buffers; the patterns only name parameters.
            if buf is not None and not isinstance(buf, DTensor):
                module.register_buffer(buf_name, distribute_tensor(buf.detach(), mesh, full_replicate))
    return model


def _patch_dtensor_conv_and_pad() -> None:
    # Both halves of WanDeviceManager._patch_dtensor_conv. The support check is forced
    # True because DTensor's TP-conv rule rejects the Wan convolutions outright, and
    # F.pad is unwrapped because DTensor's own pad drops mesh dims from the result spec
    # (a 2-D mesh comes back with a single Replicate(), and the next op fails with
    # "list index out of range"). The TorchScript guard the real patch carries is
    # irrelevant here -- nothing in a dry run is scripted.
    import torch.distributed.tensor._tp_conv as _tp_conv
    import torch.nn.functional as F

    _tp_conv._is_supported = lambda *args, **kwargs: True

    _orig_pad = F.pad

    def _pad_dtensor_safe(input, pad, mode="constant", value=None):
        if isinstance(input, DTensor):
            mesh, placements = input.device_mesh, input.placements
            out_local = _orig_pad(input.to_local(), pad, mode=mode, value=value)
            return DTensor.from_local(out_local, device_mesh=mesh, placements=placements)
        return _orig_pad(input, pad, mode=mode, value=value)

    F.pad = _pad_dtensor_safe


def _patch_tt_redistribute_constraint() -> None:
    # Mirror tt-kurbla's stricter Replicate->Shard rule (tt_kurbla/torch/_distributed.py).
    # The tt backend drives every chip from one process, so it cannot implement that
    # redistribute as "each rank keeps its own chunk"; it issues a reduce-scatter
    # instead, which needs an even split. Stock DTensor chunks unevenly without
    # complaint, so a CPU dry run would silently accept a redistribute that fails on
    # device with:
    #   ValueError: tt_kurbla.reduce_scatter: dim 2 (size 390) is not divisible by 4
    # tt-kurbla's own patch is scoped to `mesh.device_type == "tt"`, so importing it
    # would not help here -- this mesh is CPU. Hence the mirror.
    from torch.distributed.tensor.placement_types import Shard as ShardPlacement

    _orig = ShardPlacement._replicate_to_shard

    def _replicate_to_shard(self, local_tensor, mesh, mesh_dim, shard_index):
        chunks = mesh.size(mesh_dim)
        size = local_tensor.shape[self.dim]
        if size % chunks:
            raise ValueError(
                f"tt_kurbla.reduce_scatter: dim {self.dim} (size {size}) is not "
                f"divisible by the group size {chunks}"
            )
        return _orig(self, local_tensor, mesh, mesh_dim, shard_index)

    ShardPlacement._replicate_to_shard = _replicate_to_shard


def _replicated(tensor: torch.Tensor, mesh: DeviceMesh) -> DTensor:
    return distribute_tensor(tensor, mesh, [Replicate()] * mesh.ndim)


# --- stages ---------------------------------------------------------------------------


def _meta_zeros(*shape, dtype=torch.bfloat16) -> torch.Tensor:
    return torch.zeros(*shape, dtype=dtype, device="meta")


def run_dit(args, cfg, mesh, dtype):
    from diffusers import WanTransformer3DModel
    from peft import LoraConfig

    dit_config = dict(WAN22_TI2V_5B_DIT_CONFIG)
    if args.layers is not None:
        dit_config["num_layers"] = args.layers
    with torch.device("meta"):
        model = WanTransformer3DModel.from_config(dit_config)
    model = model.to(dtype).eval()
    model.add_adapter(
        LoraConfig(
            r=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            target_modules=list(cfg["lora_targets"]),
            lora_dropout=0.0,
            init_lora_weights="gaussian",
        )
    )
    for p in model.parameters():
        p.requires_grad_(False)
    shard_model(model, mesh, cfg)
    if args.backward:
        # Mirror build_lora_transformer: the base weights stay frozen and only the LoRA
        # adapters train, which is what decides where autograd puts its nodes.
        trainable = 0
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
                trainable += 1
        model.train()
        detail_suffix = f", backward over {trainable} LoRA tensors"
    else:
        detail_suffix = ""

    lat_h, lat_w = args.h // VAE_SPATIAL_STRIDE, args.w // VAE_SPATIAL_STRIDE
    p_t, p_h, p_w = dit_config["patch_size"]
    lat_f = (args.frames - 1) // VAE_TEMPORAL_STRIDE + 1
    tokens = (lat_f // p_t) * (lat_h // p_h) * (lat_w // p_w)

    hidden_states = _replicated(_meta_zeros(1, dit_config["in_channels"], lat_f, lat_h, lat_w, dtype=dtype), mesh)
    # expand_timesteps (TI2V-5B): one timestep per token, as generate.py builds it.
    timestep = _replicated(_meta_zeros(1, tokens, dtype=dtype), mesh)
    # UMT5's output is sharded on the hidden dim over "batch" -- that is what the
    # precompute cache holds and what text_embedder.linear_1 wants as input.
    encoder_hidden_states = _replicated(_meta_zeros(1, args.text_len, 4096, dtype=dtype), mesh)
    if mesh.ndim > 1 and "batch" in (mesh.mesh_dim_names or ()):
        placements = [Shard(2) if a == "batch" else Replicate() for a in mesh.mesh_dim_names]
        encoder_hidden_states = encoder_hidden_states.redistribute(mesh, placements)

    detail = f"{tokens} tokens, {dit_config['num_layers']} blocks{detail_suffix}"
    out = model(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        return_dict=False,
    )[0]
    if args.backward:
        # A scalar to differentiate. The value is meaningless on meta tensors; what is
        # being exercised is the autograd graph's placement propagation, which reaches
        # ops the forward never touches (aten.linear_backward, aten.matmul_backward).
        out.to(torch.float32).sum().backward()
    return out, detail


def run_umt5(args, cfg, mesh, dtype):
    from transformers import AutoConfig, UMT5EncoderModel

    try:
        text_config = AutoConfig.from_pretrained(cfg["model_id"], subfolder="text_encoder", local_files_only=True)
    except Exception as e:  # noqa: BLE001 - offline config is a skip, not a failure
        raise RuntimeError(f"SKIP: no cached text_encoder config ({type(e).__name__})") from e

    with torch.device("meta"):
        model = UMT5EncoderModel(text_config)
    model = model.to(dtype).eval()
    # precompute.py re-ties these after the device move; do the same so the tied weight
    # is sharded once, not twice.
    model.encoder.embed_tokens.weight = model.shared.weight
    shard_model(model, mesh, cfg)

    input_ids = _replicated(torch.zeros(1, args.text_len, dtype=torch.int64, device="meta"), mesh)
    attention_mask = _replicated(torch.ones(1, args.text_len, dtype=torch.int64, device="meta"), mesh)
    out = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    return out, f"{args.text_len} tokens, {text_config.num_layers} layers"


def _build_vae(mesh, cfg, dtype):
    from diffusers import AutoencoderKLWan

    with torch.device("meta"):
        vae = AutoencoderKLWan.from_config(WAN22_TI2V_5B_VAE_CONFIG)
    vae = vae.to(dtype).eval()
    shard_model(vae, mesh, cfg)
    return vae


def run_vae_encode(args, cfg, mesh, dtype):
    vae = _build_vae(mesh, cfg, dtype)
    video = _replicated(_meta_zeros(1, 3, args.frames, args.h, args.w, dtype=dtype), mesh)
    out = vae.encode(video).latent_dist.mode()
    return out, f"{args.frames}x{args.h}x{args.w}"


def run_vae_decode(args, cfg, mesh, dtype):
    vae = _build_vae(mesh, cfg, dtype)
    lat_f = (args.frames - 1) // VAE_TEMPORAL_STRIDE + 1
    latent = _replicated(
        _meta_zeros(1, WAN22_TI2V_5B_VAE_CONFIG["z_dim"], lat_f, args.h // VAE_SPATIAL_STRIDE,
                    args.w // VAE_SPATIAL_STRIDE, dtype=dtype),
        mesh,
    )
    out = vae.decode(latent, return_dict=False)[0]
    return out, f"{lat_f}x{args.h // VAE_SPATIAL_STRIDE}x{args.w // VAE_SPATIAL_STRIDE} latents"


RUNNERS = {"dit": run_dit, "umt5": run_umt5, "vae-encode": run_vae_encode, "vae-decode": run_vae_decode}


# --- driver ---------------------------------------------------------------------------


def _report_failure(exc: BaseException, full_traceback: bool) -> str:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if "diffusers" in f.filename
              or "transformers" in f.filename or "blacksmith" in f.filename]
    lines = str(exc).splitlines()
    print(f"    {type(exc).__name__}: {lines[0]}")
    for extra in lines[1:4]:
        if extra.strip():
            print(f"      {extra}")
    if frames:
        print("    model code involved (innermost last):")
        for f in frames[-5:]:
            print(f"      {f.filename.split('site-packages/')[-1]}:{f.lineno}  {f.line}")
    if full_traceback:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    return lines[0]


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())

    if not cfg.get("mesh_shape"):
        print(f"[dryrun] {args.config} has mesh_shape: null — no sharding to check.")
        return 0

    args.h = args.h if args.h is not None else cfg["train_h"]
    args.w = args.w if args.w is not None else cfg["train_w"]
    dtype = getattr(torch, str(cfg.get("dtype", "torch.bfloat16")).removeprefix("torch."))

    mesh_shape = tuple(cfg["mesh_shape"])
    axis_names = tuple(cfg["mesh_axis_names"])
    world = int(torch.tensor(mesh_shape).prod())

    from torch.testing._internal.distributed.fake_pg import FakeStore

    dist.init_process_group("fake", rank=0, world_size=world, store=FakeStore())
    mesh = DeviceMesh("cpu", torch.arange(world).reshape(mesh_shape), mesh_dim_names=axis_names)
    _patch_dtensor_conv_and_pad()
    _patch_tt_redistribute_constraint()

    if not args.no_overrides:
        from blacksmith.models.torch.wan2_2.model_overrides import apply_generality_overrides

        apply_generality_overrides()

    print(f"[dryrun] {Path(args.config).name}: mesh {dict(zip(axis_names, mesh_shape))} = {world} ranks, "
          f"{args.frames}x{args.h}x{args.w}, {dtype}"
          f"{'' if not args.no_overrides else ', overrides DISABLED'}")

    stages = STAGES if args.stage == "all" else (args.stage,)
    results: dict[str, dict] = {}
    for stage in stages:
        start = time.perf_counter()
        try:
            out, detail = RUNNERS[stage](args, cfg, mesh, dtype)
        except Exception as exc:  # noqa: BLE001 - reporting the failure is the job
            elapsed = time.perf_counter() - start
            message = str(exc).splitlines()[0]
            if message.startswith("SKIP: "):
                print(f"  {stage:12s} SKIP  {message.removeprefix('SKIP: ')}")
                results[stage] = {"status": "skip", "reason": message, "seconds": elapsed}
                continue
            print(f"  {stage:12s} FAIL  ({elapsed:.1f}s)")
            results[stage] = {"status": "fail", "error": _report_failure(exc, args.full_traceback),
                              "seconds": elapsed}
            continue
        elapsed = time.perf_counter() - start
        placements = tuple(str(p) for p in out.placements) if isinstance(out, DTensor) else ("plain tensor",)
        print(f"  {stage:12s} PASS  ({elapsed:.1f}s)  {detail} -> {tuple(out.shape)} {placements}")
        results[stage] = {"status": "pass", "shape": list(out.shape), "placements": list(placements),
                          "seconds": elapsed}

    failed = [s for s, r in results.items() if r["status"] == "fail"]
    print(f"\n[dryrun] {len(results) - len(failed)}/{len(results)} stages passed"
          + (f"; failed: {', '.join(failed)}" if failed else ""))

    if args.json:
        Path(args.json).write_text(json.dumps({"config": args.config, "mesh": list(mesh_shape),
                                               "stages": results}, indent=2))
        print(f"[dryrun] wrote {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
