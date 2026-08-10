# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Mesh bring-up ladder for Wan 2.2 on tt-kurbla — the multichip counterpart of `bringup.py`.

`bringup.py` proves one component on one chip. This proves one component on a *mesh*, one
rung at a time, and writes a JSON artifact per run so a result can be re-checked without
re-reading a log.

    P=blacksmith.experiments.torch.wan2_2.kurbla.mesh_ladder
    KPY=~/tt-kurbla/venv/bin/python

    $KPY -m $P --rung replicate --layers 1                      # plumbing
    $KPY -m $P --rung replicate --layers 30 --h 480 --w 832     # real shape, all layers
    $KPY -m $P --rung replicate --layers 30 --pretrained        # real weights
    $KPY -m $P --rung dp        --layers 30 --batch 4           # data parallel
    $KPY -m $P --rung tp        --layers 30                     # tensor parallel (quietbox patterns)
    $KPY -m $P --rung dp --lora --backward --no-check           # a training step

**Mesh shape.** Defaults to the full chip count as a 2-D mesh, because on an 8x4 Blackhole
galaxy *only* the full 32-chip 2-D mesh works -- every submesh dies in fabric router sync and
a 1-D 32-chip mesh hangs. See CONTEXT-2.md. Do not "simplify" this to [1,2] to debug.

**Always keep `--strict`** (on by default here): without it an unimplemented op runs on the
host and still yields an excellent pcc.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import yaml

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rung", choices=("replicate", "dp", "tp"), default="replicate",
                   help="replicate = null sharding; dp = shard the batch; tp = quietbox model patterns.")
    p.add_argument("--stage", choices=("dit", "vae-encode", "vae-decode"), default="dit")
    p.add_argument("--layers", type=int, default=1, help="DiT blocks (30 = full model).")
    p.add_argument("--h", type=int, default=64)
    p.add_argument("--w", type=int, default=64)
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--text-len", type=int, default=32)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    p.add_argument("--mode", choices=("eager", "compile"), default="compile")
    p.add_argument("--mesh", default=None, help="ROWSxCOLS; default = balanced 2-D over all chips.")
    p.add_argument("--pretrained", action="store_true", help="Real TI2V-5B weights.")
    p.add_argument("--lora", action="store_true")
    p.add_argument("--backward", action="store_true", help="Flow-matching loss + backward.")
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--steps", type=int, default=0,
                   help="Optimizer steps (implies --lora --backward --no-check): a real training loop.")
    p.add_argument("--tolerance", type=float, default=0.98)
    p.add_argument("--no-check", action="store_true", help="Skip the CPU reference (it is slow at 30 layers).")
    p.add_argument("--no-strict", dest="strict", action="store_false", help="DANGEROUS: allow CPU fallbacks.")
    p.add_argument("--artifacts", default=str(Path.home() / "wan22-kurbla" / "mesh-runs"))
    p.add_argument("--tag", default=None, help="Artifact filename tag; default derived from the config.")
    p.set_defaults(strict=True)
    return p.parse_args(argv)


def default_mesh(n: int) -> tuple[int, int]:
    """Balanced 2-D (rows, cols) with rows <= cols — the shape the fabric actually supports."""
    rows = 1
    for d in range(1, math.isqrt(n) + 1):
        if n % d == 0:
            rows = d
    return rows, n // rows


def _git_rev(path: str) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def build_config(args, mesh_shape):
    """A TrainingConfig carrying just the device/mesh fields the manager reads."""
    from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig
    from blacksmith.tools.cli import generate_config
    from blacksmith.experiments.torch.wan2_2.kurbla import bringup as B

    here = Path(B.__file__).parent
    config = generate_config(TrainingConfig, here / "lora" / "single_chip" / "wan2_2_ti2v_5b_diffusiondb.yaml", None, None)
    config.use_tt = True
    config.mesh_shape = list(mesh_shape)
    config.mesh_axis_names = ["batch", "model"]

    if args.rung == "replicate":
        config.input_sharding_dim = None
        config.model_sharding_patterns = []
    elif args.rung == "dp":
        config.input_sharding_dim = "batch"
        config.model_sharding_patterns = []
    else:  # tp — reuse the patterns the tt-xla quietbox config already ships
        quietbox = here / "lora" / "quietbox" / "wan2_2_ti2v_5b_diffusiondb.yaml"
        patterns = yaml.safe_load(quietbox.read_text())["model_sharding_patterns"]
        config.input_sharding_dim = None
        config.model_sharding_patterns = patterns
    return config


def main(argv=None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)
    wall = time.perf_counter()

    from blacksmith.models.torch.wan2_2.model_overrides import apply_generality_overrides, apply_perf_overrides

    apply_generality_overrides()
    apply_perf_overrides()

    from blacksmith.experiments.torch.wan2_2.kurbla import bringup as B
    from blacksmith.experiments.torch.wan2_2.kurbla.model_overrides import apply_kurbla_overrides

    # Reuse bringup's model/input builders verbatim so a mesh result is comparable to the
    # single-chip number for the same flags.
    bringup_flags = ["--stage", args.stage, "--layers", str(args.layers), "--h", str(args.h),
                     "--w", str(args.w), "--frames", str(args.frames), "--text-len", str(args.text_len),
                     "--batch", str(args.batch), "--dtype", args.dtype, "--mode", args.mode]
    if args.pretrained:
        bringup_flags.append("--pretrained")
    bargs = B.parse_args(bringup_flags)

    is_vae = args.stage.startswith("vae")
    base = B.build_vae(bargs) if is_vae else B.build_dit(bargs)
    rewrites = apply_kurbla_overrides(base)
    print(f"[ladder] kurbla rewrites: {rewrites}", flush=True)

    inputs = B.make_vae_inputs(bargs, base) if is_vae else B.make_inputs(bargs, base)
    module = (B._vae_stages() if is_vae else B._STAGES)[args.stage](base)

    if args.lora:
        from peft import LoraConfig

        assert not is_vae, "--lora targets the DiT projections"
        for prm in base.parameters():
            prm.requires_grad_(False)
        base.add_adapter(LoraConfig(r=32, lora_alpha=32,
                                    target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
                                    lora_dropout=0.0, init_lora_weights="gaussian"))

    n_params = sum(p.numel() for p in module.parameters())
    print(f"[ladder] stage={args.stage} rung={args.rung} layers={args.layers} params={n_params/1e6:.1f}M "
          f"pretrained={args.pretrained} lora={args.lora} backward={args.backward} mode={args.mode}", flush=True)
    for name, tensor in inputs.items():
        print(f"[ladder]   {name}: {tuple(tensor.shape)} {tensor.dtype}", flush=True)

    reference = None
    if not args.no_check:
        t = time.perf_counter()
        with torch.no_grad():
            reference = module(**inputs)
        print(f"[ladder] cpu reference {tuple(reference.shape)} in {time.perf_counter()-t:.1f}s", flush=True)

    import tt_kurbla.torch  # noqa: F401 — registers the tt device / dynamo backend

    chips = torch.tt.num_chips()
    mesh_shape = tuple(int(x) for x in args.mesh.split("x")) if args.mesh else default_mesh(chips)
    print(f"[ladder] chips={chips} mesh={mesh_shape}", flush=True)

    config = build_config(args, mesh_shape)
    from blacksmith.experiments.torch.wan2_2.kurbla.device_manager import WanDeviceManager

    dm = WanDeviceManager(config)
    print(f"[ladder] device_mesh={dm.mesh} dp={dm.is_data_parallel()} tp={dm.is_tensor_parallel()} "
          f"patterns={len(config.model_sharding_patterns)}", flush=True)

    t = time.perf_counter()
    module = dm.shard_model(module)
    module = dm.to_device(module)
    shard_s = time.perf_counter() - t
    # prepare_batch (not to_device): it wraps inputs as DTensors, and DTensor refuses to mix
    # a plain tensor with the replicated/sharded parameters.
    inputs = dm.prepare_batch(inputs)
    # A sharding pattern that matches nothing leaves the model replicated and still scores a
    # perfect pcc -- the same class of false PASS as the degenerate-tensor trap. Count what
    # actually got sharded so a TP run cannot silently be a replicate run.
    sharded_params, replicated_params = 0, 0
    for prm in module.parameters():
        if hasattr(prm, "placements"):
            if any(pl.is_shard() for pl in prm.placements):
                sharded_params += 1
            else:
                replicated_params += 1
    print(f"[ladder] sharded + moved to device in {shard_s:.1f}s "
          f"(params: {sharded_params} sharded / {replicated_params} replicated)", flush=True)

    if args.mode == "compile":
        module = dm.compile(module)

    def make_guard():
        """A fresh guard per use: `strict_no_fallback()` is a one-shot generator CM."""
        if not args.strict:
            return nullcontext()
        from tt_kurbla.torch.testing import strict_no_fallback

        return strict_no_fallback()

    out = None
    times = []
    with make_guard():
        for i in range(args.iters):
            t = time.perf_counter()
            with torch.enable_grad() if args.backward else torch.no_grad():
                out = module(**inputs)
            times.append(time.perf_counter() - t)
            print(f"[ladder] forward {i}: {tuple(out.shape)} in {times[-1]:.2f}s", flush=True)

    # --- optimizer loop: the actual finetuning path -------------------------------------
    if args.steps:
        trainable = [p_ for p_ in module.parameters() if p_.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01, foreach=False)
        probe = trainable[0]
        before = probe.to_local().clone() if hasattr(probe, "to_local") else probe.detach().clone()
        losses = []
        for step in range(args.steps):
            t = time.perf_counter()
            # Strict covers the model only. The AdamW step itself still falls back to the
            # host: tt-kurbla has no lowering for the out-variant `aten::mul.out`, and
            # `foreach=False` does not avoid it. So the optimizer math runs on CPU today —
            # recorded in the artifact as `optimizer_on_host` rather than hidden by
            # dropping --strict, which would also mask fallbacks in the model.
            with make_guard():
                out = module(**inputs)
                target = torch.zeros_like(out)
                loss = torch.nn.functional.mse_loss(out.float(), target.float())
                loss.backward()
            dm.optimizer_step(optimizer, zero_grad=True)
            losses.append(float(loss.item()))
            print(f"[ladder] step {step}: loss={losses[-1]:.6f} in {time.perf_counter()-t:.1f}s", flush=True)
        after = probe.to_local() if hasattr(probe, "to_local") else probe.detach()
        delta = (after.float() - before.float()).abs().max().item()
        print(f"[ladder] weights moved: max|delta|={delta:.3e} over {args.steps} steps "
              f"(loss {losses[0]:.6f} -> {losses[-1]:.6f})", flush=True)
        record_extra = {"losses": losses, "weight_max_delta": delta, "n_trainable": len(trainable),
                        "optimizer_on_host": True, "optimizer_host_reason": "aten::mul.out not lowered"}
        assert delta > 0, "optimizer ran but no trainable weight changed"
    else:
        record_extra = {}

    loss_value = None
    n_grads = None
    if args.backward and not args.steps:
        target = torch.zeros_like(out)
        loss = torch.nn.functional.mse_loss(out.float(), target.float())
        loss.backward()
        grads = [p for p in module.parameters() if p.grad is not None]
        loss_value, n_grads = float(loss.item()), len(grads)
        print(f"[ladder] backward ok: loss={loss_value:.6f}, {n_grads} params with grad", flush=True)

    score = note = None
    passed = True
    if reference is not None:
        gathered = dm.gather(out) if hasattr(out, "_spec") else out
        score, note = B.compare(gathered.cpu().float(), reference.float())
        passed = score >= args.tolerance
        print(f"[ladder] pcc={score:.5f} vs cpu (tolerance {args.tolerance}) {note} -> "
              f"{'PASS' if passed else 'FAIL'}", flush=True)
    else:
        print("[ladder] no CPU reference (--no-check): ran to completion", flush=True)

    record = {
        "started_utc": started.isoformat(),
        "wall_s": round(time.perf_counter() - wall, 1),
        "host": platform.node(),
        "chips": chips,
        "mesh_shape": list(mesh_shape),
        "rung": args.rung,
        "sharding_patterns": len(config.model_sharding_patterns),
        "input_sharding_dim": config.input_sharding_dim,
        "stage": args.stage,
        "layers": args.layers,
        "shape": {"h": args.h, "w": args.w, "frames": args.frames, "text_len": args.text_len, "batch": args.batch},
        "dtype": args.dtype,
        "mode": args.mode,
        "pretrained": args.pretrained,
        "lora": args.lora,
        "backward": args.backward,
        "strict": args.strict,
        "params_millions": round(n_params / 1e6, 1),
        "rewrites": rewrites,
        "shard_and_move_s": round(shard_s, 1),
        "params_sharded": sharded_params,
        "params_replicated": replicated_params,
        "forward_s": [round(x, 2) for x in times],
        "pcc": None if score is None else round(float(score), 5),
        "pcc_note": note,
        "loss": loss_value,
        "params_with_grad": n_grads,
        "passed": bool(passed),
        **record_extra,
        "tt_kurbla_rev": _git_rev(str(Path.home() / "tt-kurbla")),
        "tt_blacksmith_rev": _git_rev(str(Path.home() / "tt-blacksmith")),
    }
    tag = args.tag or (f"{args.stage}-{args.rung}-L{args.layers}-{args.h}x{args.w}f{args.frames}"
                       f"-{args.mode}{'-pretrained' if args.pretrained else ''}"
                       f"{'-lora' if args.lora else ''}{'-bwd' if args.backward else ''}")
    out_dir = Path(args.artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"[ladder] artifact -> {path}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
