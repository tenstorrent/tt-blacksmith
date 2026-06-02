"""Recompute the Wan 2.2 5B precompute (VAE latents + UMT5 embeds) on CPU and
PCC-compare against the existing TT-generated cache.

Goal: decide whether the TT precompute cache (cache/wan22_5b) is corrupted
relative to a correct, same-dtype CPU forward. If PCC is high (>~0.99) the
cache is fine and the training-loss problem is elsewhere (e.g. the scalar
timestep DiT graph); if PCC is low the precompute itself is the bug.

We deliberately import the dataset / preprocessing helpers from `tmp.py`
(the GPU recipe) instead of `wan22_5b.py`, because the latter applies the TT
overrides and pulls in torch_xla at import time. The dataset selection is
deterministic (seeded shuffle), so sample_i here is the same image that
produced cache/wan22_5b/samples/sample_i.pt on TT.

Run from the tt-blacksmith dir (same cwd the TT precompute used, so the
relative CACHE_DIR resolves):

    python compare_precompute_cpu.py
    python compare_precompute_cpu.py --dtype float32   # fp32 golden instead
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tmp import (
    Config,
    _pil_to_video_tensor,
    _wan_latents_normalize,
    download_and_subset_dataset,
)

from diffusers import AutoencoderKLWan
from transformers import AutoTokenizer, UMT5EncoderModel


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to("cpu", torch.float32).flatten()
    b = b.detach().to("cpu", torch.float32).flatten()
    if a.shape != b.shape:
        return float("nan")
    va, vb = a - a.mean(), b - b.mean()
    denom = va.norm() * vb.norm()
    if denom == 0:
        return float("nan")
    return float((va @ vb) / denom)


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to("cpu", torch.float32)
    b = b.detach().to("cpu", torch.float32)
    return float((a - b).norm() / (b.norm() + 1e-12))


def summarize(name: str, pccs: list[float]) -> None:
    import numpy as np

    arr = np.asarray([p for p in pccs if p == p])  # drop nan
    n_nan = len(pccs) - len(arr)
    if len(arr) == 0:
        print(f"[{name}] no valid comparisons (all nan / shape mismatch)")
        return
    print(
        f"[{name}] n={len(pccs)} (nan/shape-mismatch={n_nan})  "
        f"PCC min={arr.min():.4f} mean={arr.mean():.4f} median={np.median(arr):.4f} "
        f"max={arr.max():.4f}  | #below0.99={(arr < 0.99).sum()} #below0.9={(arr < 0.9).sum()}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                    help="CPU forward dtype. bfloat16 matches the TT precompute dtypes.")
    ap.add_argument("--max-samples", type=int, default=0, help="limit #latents compared (0=all)")
    ap.add_argument("--cache-dir", type=str, default=None, help="override CACHE_DIR")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    cfg = Config()
    cfg.DEVICE = "cpu"
    cache = Path(args.cache_dir or cfg.CACHE_DIR)
    samples_dir = cache / "samples"
    embeds_path = cache / "embeds.pt"
    meta_path = cache / "metadata.json"
    for p in (samples_dir, embeds_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p} — run from the dir holding {cfg.CACHE_DIR}")

    metadata = json.loads(meta_path.read_text())
    idx_to_caption = {int(m["idx"]): m["caption"] for m in metadata}
    print(f"[cache] {cache.resolve()}  samples={len(metadata)}  dtype(cpu)={args.dtype}")

    # --- Re-derive the exact same image subset (deterministic seed) ---
    print("[data] re-deriving subset on CPU (uses cached HF downloads) ...")
    samples = download_and_subset_dataset(cfg)
    print(f"[data] got {len(samples)} samples")

    # --- VAE latents ---
    print(f"[vae] loading AutoencoderKLWan (cpu, {args.dtype}) ...")
    vae = AutoencoderKLWan.from_pretrained(
        cfg.MODEL_ID, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval()

    n = len(samples) if args.max_samples <= 0 else min(args.max_samples, len(samples))
    lat_pccs: list[float] = []
    cap_mismatch = 0
    print(f"[vae] encoding + comparing {n} latents ...")
    with torch.no_grad():
        for i in range(n):
            img, raw_caption = samples[i]
            triggered = cfg.TRIGGER + raw_caption.strip()
            if idx_to_caption.get(i) != triggered:
                cap_mismatch += 1
                if cap_mismatch <= 3:
                    print(f"  [warn] caption mismatch at idx {i}: dataset/order may differ "
                          f"from the cached run -> latent comparison for this idx is meaningless")
            video = _pil_to_video_tensor(img, cfg.TRAIN_H, cfg.TRAIN_W).to(dtype)
            latent = vae.encode(video).latent_dist.mode()
            latent = _wan_latents_normalize(latent, vae).squeeze(0)  # (C,F,H,W)

            cached = torch.load(samples_dir / f"sample_{i:04d}.pt", weights_only=False)["latent"]
            p = pcc(latent, cached)
            lat_pccs.append(p)
            if i < 5 or p < 0.99:
                print(f"  latent[{i:04d}] pcc={p:.4f} rel_err={rel_err(latent, cached):.4f} "
                      f"cpu_shape={tuple(latent.shape)} cache_shape={tuple(cached.shape)}")

    if cap_mismatch:
        print(f"[warn] {cap_mismatch}/{n} captions did not match the cache -> dataset order "
              f"drifted; treat latent PCC with suspicion.")
    summarize("vae-latent", lat_pccs)

    del vae

    # --- UMT5 text embeds ---
    print(f"[umt5] loading UMT5EncoderModel (cpu, {args.dtype}) ...")
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_ID, subfolder="tokenizer")
    enc = UMT5EncoderModel.from_pretrained(
        cfg.MODEL_ID, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True
    ).eval()
    # Same tie the TT precompute does (here on CPU it cannot be broken by a device move).
    enc.encoder.embed_tokens.weight = enc.shared.weight

    cached_embeds = torch.load(embeds_path, weights_only=False)
    unique_caps = sorted({m["caption"] for m in metadata})
    if "" in cached_embeds and "" not in unique_caps:
        unique_caps.append("")

    emb_pccs: list[float] = []
    print(f"[umt5] encoding + comparing {len(unique_caps)} unique captions ...")
    with torch.no_grad():
        for j, cap in enumerate(unique_caps):
            if cap not in cached_embeds:
                print(f"  [warn] caption not in cached embeds.pt: {cap[:40]!r}")
                continue
            t = tok(cap, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            out = enc(input_ids=t.input_ids, attention_mask=t.attention_mask).last_hidden_state
            out = out * t.attention_mask.unsqueeze(-1).to(out.dtype)
            out = out.squeeze(0)
            p = pcc(out, cached_embeds[cap])
            emb_pccs.append(p)
            if j < 5 or p < 0.99:
                print(f"  embed[{j:03d}] pcc={p:.4f} rel_err={rel_err(out, cached_embeds[cap]):.4f} "
                      f"seq={int(t.attention_mask.sum())} cap={cap[:40]!r}")

    summarize("umt5-embed", emb_pccs)
    print("\n[done] High PCC (>~0.99) => TT precompute cache is faithful; problem is in "
          "training (e.g. scalar-timestep DiT). Low PCC => precompute itself is the bug.")


if __name__ == "__main__":
    main()
