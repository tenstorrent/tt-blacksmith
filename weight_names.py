"""Print every parameter name + shape + dtype for the three Wan 2.2 TI2V-5B
sub-models (UMT5 text encoder, AutoencoderKLWan VAE, WanTransformer3DModel
DiT).

Used as a reference when authoring LoRA target patterns and SPMD shard specs
in `override.py` / `wan22_5b.py`.

Run with:
    python weight_names.py                    # all three, summary + full dump
    python weight_names.py --no-dump          # summary only (counts + size)
    python weight_names.py --model dit        # just the DiT, etc.
    python weight_names.py --grep attn1.to_q  # filter dumped names by substring
"""

from __future__ import annotations

import argparse
from typing import Iterable

import torch
import torch.nn as nn

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def _fmt_shape(t: torch.Tensor) -> str:
    return "x".join(str(d) for d in t.shape) if t.ndim > 0 else "scalar"


def _iter_params(module: nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    """Named parameters + named buffers, in declaration order, deduplicated."""
    seen: set[int] = set()
    for name, p in module.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        yield name, p
    for name, b in module.named_buffers():
        if id(b) in seen:
            continue
        seen.add(id(b))
        yield name, b


def _print_section(title: str, module: nn.Module, dump: bool, grep: str | None) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)

    total_params = 0
    total_bytes = 0
    n = 0
    n_trainable = 0
    rows: list[tuple[str, str, str, int]] = []

    for name, t in _iter_params(module):
        numel = t.numel()
        total_params += numel
        total_bytes += numel * t.element_size()
        n += 1
        if t.requires_grad:
            n_trainable += 1
        rows.append((name, _fmt_shape(t), str(t.dtype).replace("torch.", ""), numel))

    print(
        f"  tensors={n}  trainable={n_trainable}  "
        f"params={total_params / 1e6:.2f}M  size={total_bytes / 2**20:.1f} MiB"
    )

    if not dump:
        return

    if grep:
        rows = [r for r in rows if grep in r[0]]
        print(f"  (filtered by --grep {grep!r}: {len(rows)} tensors)")

    if not rows:
        return

    name_w = max(len(r[0]) for r in rows)
    shape_w = max(len(r[1]) for r in rows)
    dtype_w = max(len(r[2]) for r in rows)
    print()
    for name, shape, dtype, numel in rows:
        print(
            f"  {name.ljust(name_w)}  {shape.ljust(shape_w)}  "
            f"{dtype.ljust(dtype_w)}  numel={numel}"
        )


def dump_umt5(dump: bool, grep: str | None) -> None:
    from transformers import UMT5EncoderModel

    print("[load] UMT5EncoderModel (text_encoder) ...")
    enc = UMT5EncoderModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    enc.encoder.embed_tokens.weight = enc.shared.weight
    _print_section("UMT5EncoderModel (text_encoder)", enc, dump, grep)


def dump_vae(dump: bool, grep: str | None) -> None:
    from diffusers import AutoencoderKLWan

    print("[load] AutoencoderKLWan (vae) ...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    _print_section("AutoencoderKLWan (vae) — full", vae, dump, grep)
    _print_section("AutoencoderKLWan.encoder", vae.encoder, dump, grep)
    _print_section("AutoencoderKLWan.decoder", vae.decoder, dump, grep)
    if hasattr(vae, "quant_conv") and vae.quant_conv is not None:
        _print_section("AutoencoderKLWan.quant_conv", vae.quant_conv, dump, grep)
    if hasattr(vae, "post_quant_conv") and vae.post_quant_conv is not None:
        _print_section(
            "AutoencoderKLWan.post_quant_conv", vae.post_quant_conv, dump, grep
        )


def dump_dit(dump: bool, grep: str | None) -> None:
    from diffusers import WanTransformer3DModel

    print("[load] WanTransformer3DModel (transformer) ...")
    dit = WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()
    _print_section("WanTransformer3DModel (transformer)", dit, dump, grep)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        choices=["all", "umt5", "vae", "dit"],
        default="all",
        help="which sub-model to dump",
    )
    p.add_argument(
        "--no-dump",
        action="store_true",
        help="only print per-section summary (counts + size), skip full name list",
    )
    p.add_argument(
        "--grep",
        type=str,
        default=None,
        help="substring filter applied to parameter names before dumping",
    )
    args = p.parse_args()

    dump = not args.no_dump

    if args.model in ("all", "umt5"):
        dump_umt5(dump, args.grep)
    if args.model in ("all", "vae"):
        dump_vae(dump, args.grep)
    if args.model in ("all", "dit"):
        dump_dit(dump, args.grep)


if __name__ == "__main__":
    main()
