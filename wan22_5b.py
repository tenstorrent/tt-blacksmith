"""LoRA fine-tune of Wan 2.2 TI2V-5B on jainr3/diffusiondb-pixelart (Tenstorrent).

TT port of `tmp.py`. Same config / logging / training loop, with minimal
TT-specific changes:

  * `override.apply_generality_overrides()` + `apply_perf_overrides()` applied
    at import time (before any model import that diffusers might cache the
    decorated `WanTransformer3DModel.forward` from).
  * Device management goes through `override.WanDeviceManager`:
      - SPMD mesh = (2, 4) ("batch", "model") on 8 devices.
      - `torch_xla.set_custom_compile_options` (opt level 1, fp32 acc, hifi4,
        DRAM space-saving — same knobs as ppadjin/wan5b_tests)
      - `torch.compile(model, backend="tt",
        options={"tt_enable_torch_fx_fusion_pass": False,
                 "tt_legacy_compile": True})` for every model that gets a
        forward pass on TT (text encoder, VAE encoder, VAE decoder, DiT).
  * Data preprocessing (PIL / parquet / zip / center-crop / `_pil_to_video_tensor`)
    stays on CPU; only the model forward passes move to TT.
  * Noise + timestep generation stays on CPU for training, validation, and
    inference (CPU `torch.Generator`, then `.to(device)`).
  * VAE-decoder-touching paths (validation sample + final inference) are
    wrapped in `override.safe_xla_slicing()` — `AutoencoderKLWan` relies on
    CPU's silent slice-clamping which torch-xla rejects.
"""

from __future__ import annotations

import chisel

import argparse
import dataclasses
import gc
import io
import json
import math
import os
import random
import time
import zipfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Overrides MUST run before the diffusers imports below: `_patch_apply_lora_scale`
# unwraps a class attribute that diffusers binds at import time, and
# `_patch_wan_resample_avoid_4d_fold` swaps in a new `WanResample.forward`.
from override import (
    WanDeviceManager,
    apply_generality_overrides,
    apply_perf_overrides,
    safe_xla_slicing,
)

import torch_xla.core.xla_model as xm

apply_generality_overrides()
apply_perf_overrides()

from diffusers import (  # noqa: E402
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanPipeline,
    WanTransformer3DModel,
)
from diffusers.utils import export_to_video  # noqa: E402
from peft import LoraConfig, get_peft_model_state_dict  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoTokenizer, UMT5EncoderModel  # noqa: E402

try:
    import wandb  # noqa: F401
    _WANDB_OK = True
except Exception:
    _WANDB_OK = False


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------


@dataclass
class Config:
    # Model + device
    MODEL_ID: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    # DEVICE is not user-configurable on TT — the actual XLA device handle
    # is read from `DEVMGR.device` at runtime. Kept as a string label for
    # log lines + W&B config dump.
    DEVICE: str = "tt"
    DTYPE: Any = torch.bfloat16
    # VAE encode runs in bf16 on TT (the wan5b component tests in tt-xla
    # use bf16 throughout). The GPU path used fp32 for VAE precompute to
    # work around per-GPU bf16 mode() NaNs; not relevant here.
    VAE_PRECOMPUTE_DTYPE: Any = torch.bfloat16
    GRADIENT_CHECKPOINTING: bool = False       # off on TT (interacts badly with XLA tracing)

    # Debug / bring-up: keep only the first N UMT5 layers / DiT blocks to
    # shrink compile time. 0 = full model. Set e.g. 8 for a partial run.
    # NOTE: embeds.pt is precomputed with UMT5 at DEBUG_UMT5_LAYERS, so keep
    # this consistent between `precompute` and `train`/`infer`.
    DEBUG_UMT5_LAYERS: int = 2
    DEBUG_DIT_BLOCKS: int = 2

    # Dataset / cache
    DATASET_ID: str = "jainr3/diffusiondb-pixelart"
    DATASET_CONFIG: str = "2k_random_1k"
    CACHE_DIR: str = "cache/wan22_5b"
    SUBSET_SIZE: int = 64
    VAL_HOLDOUT: int = 4                       # 4 held-out images for val/loss
    SEED: int = 42

    # Train resolution (single image as 1-frame video).
    # Must be a multiple of 32 (= VAE spatial stride 16 * transformer patch_size 2)
    # so the round-trip latent->patch->unpatch is shape-exact.
    # 832x480 is an official Wan2.2 TI2V-5B 480p landscape bucket — better
    # baseline quality than square 480x480 (which is off-distribution).
    TRAIN_H: int = 480
    TRAIN_W: int = 832
    TRAIN_FRAMES: int = 1

    # Inference (kept equal to train res so val and final inference both run
    # the model at the same scale; pxa is sensitive to off-bucket resolution).
    INFER_H: int = 480
    INFER_W: int = 832
    INFER_FRAMES: int = 65          # 4k+1 -> 16fps ~= 4s
    INFER_FPS: int = 16
    INFER_STEPS: int = 40                  # match VAL_IMG_STEPS so val and final are apples-to-apples
    INFER_GUIDANCE: float = 5.0
    INFER_FLOW_SHIFT: float = 5.0   # Wan 2.2 TI2V-5B official default
    INFER_OUTPUT: str = "cache/wan22_5b/pixelart_video.mp4"

    # Style trigger + CFG dropout
    TRIGGER: str = "pxa, "
    TEXT_DROP_PROB: float = 0.10

    # LoRA. Covers self-attn (attn1.*), cross-attn (attn2.*), and the FFN
    # (ffn.net.0.proj, ffn.net.2) so the LoRA can adapt both text routing AND
    # visual feature mixing. PEFT does suffix-matching; these strings cover
    # both attn1 and attn2 q/k/v/out projections.
    LORA_RANK: int = 32
    LORA_ALPHA: int = 32                       # scale = alpha/r = 1.0
    LORA_TARGETS: tuple = (
        "to_q", "to_k", "to_v", "to_out.0",
        "ffn.net.0.proj", "ffn.net.2",
    )

    # Optimizer / schedule
    LR: float = 1e-4
    WEIGHT_DECAY: float = 0.01                 # gentle L2; delays overfitting
    BATCH: int = 1
    GRAD_ACCUM: int = 4
    MAX_STEPS: int = 3000                      # GPU is fast; aim for ~60 epochs
    LR_WARMUP_FRAC: float = 0.05               # 5% linear warmup, then cosine to 0

    # Flow-matching training
    TRAIN_FLOW_SHIFT: float = 3.0
    LOGNORM_MEAN: float = 0.0
    LOGNORM_STD: float = 1.0

    # Validation
    VAL_LOSS_EVERY: int = 0                    # 0 = disabled. Flow-matching MSE is a poor
                                               # style-quality signal anyway; rely on val/sample_video.
    VAL_IMG_EVERY: int = 300                   # video gen is slow; also gates ckpt save
    VAL_PROMPT: str = "a car driving through the desert with sunset in background"
    VAL_IMG_STEPS: int = 40                    # 20 → 40: ~2x val time, much cleaner samples
    VAL_IMG_FRAMES: int = 65                   # match INFER_FRAMES (~4 s @ 16 fps); 4k+1 required
    # Inherited lesson from the 14B run: Wan's default negative prompt
    # actively suppresses "style/painting/low quality" — i.e. exactly the
    # descriptors pxa produces. Default to empty here; can be overridden.
    NEG_PROMPT: str = ""

    # W&B
    WANDB_PROJECT: str = "wan22-pixelart-lora"
    WANDB_ENABLED: bool = True

    # Output
    LORA_PATH: str = "cache/wan22_5b/wan22_5b_pxa_lora.safetensors"

    def asdict(self) -> dict:
        d = asdict(self)
        d["DTYPE"] = str(self.DTYPE)
        d["LORA_TARGETS"] = list(self.LORA_TARGETS)
        return d

    def __post_init__(self):
        # vae_scale_factor_spatial(16) * transformer.patch_size[1..2](2) = 32.
        # If H or W is not a multiple of 32, the transformer's patch_embedding
        # rounds down and unpatchify produces a smaller output than the input,
        # so the MSE target/pred shapes mismatch. Catch it early.
        for label, v in [("TRAIN_H", self.TRAIN_H), ("TRAIN_W", self.TRAIN_W),
                         ("INFER_H", self.INFER_H), ("INFER_W", self.INFER_W)]:
            if v % 32 != 0:
                raise ValueError(
                    f"{label}={v} must be a multiple of 32 (VAE_stride * patch_size). "
                    f"Try {v - v % 32} or {v + 32 - v % 32}."
                )
        if (self.INFER_FRAMES - 1) % 4 != 0:
            raise ValueError(
                f"INFER_FRAMES={self.INFER_FRAMES} must satisfy 4k+1 (Wan VAE "
                f"temporal stride is 4). Closest: {self.INFER_FRAMES - (self.INFER_FRAMES - 1) % 4}."
            )


CFG = Config()


# ---------------------------------------------------------------------------
# Device manager (single instance for the whole run).
# Mirrors `tests/torch/models/wan5b/shared.run_component`:
#   - sets `xr.set_device_type("TT")` + `xm.xla_device()`
#   - sets `torch_xla.set_custom_compile_options` (fp32 acc, hifi4, DRAM
#     space-saving — same knobs as ppadjin/wan5b_tests)
#   - builds the (2,4) "batch"/"model" SPMD mesh (auto-sized on N devices)
#   - exposes `to_device`, `shard_module(inner, "component")`, `compile(m)`
#     using `torch.compile(backend="tt",
#                          options={"tt_enable_torch_fx_fusion_pass": False,
#                                   "tt_legacy_compile": True})`
# ---------------------------------------------------------------------------


DEVMGR: WanDeviceManager | None = None


def _devmgr() -> WanDeviceManager:
    global DEVMGR
    if DEVMGR is None:
        DEVMGR = WanDeviceManager(use_tt=True, sharded=False)  # DEBUG: sharding off
    return DEVMGR


# Wrappers reused by the wan5b component tests in tt-xla. They strip
# diffusers return-object types down to a plain tensor so dynamo can trace
# the call without graph-breaking on the dict-like output.


class _UMT5Wrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


class _VAEEncoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return self.vae.encode(x).latent_dist.mode()


class _WanDiTWrapper(nn.Module):
    """Plain tensor out — matches dev_wan / tt-xla wan2_2 component tests."""

    def __init__(self, dit: WanTransformer3DModel):
        super().__init__()
        self.dit = dit

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class _VAEDecoderWrapper(nn.Module):
    """Run the decoder and return the reconstructed sample tensor.

    Compiling this (vs calling `vae.decode` eagerly under lazy tensors) is
    essential: `AutoencoderKLWan._decode` loops frame-by-frame over the
    latent temporal axis with a `feat_cache`. Eager LTC traces+compiles
    *each* frame/chunk iteration separately -> dozens of recompiles. Under
    `torch.compile` dynamo unrolls the whole loop into a single graph that
    compiles once (this is what ppadjin/wan5b_tests and mstojkovic/dev_wan do).
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


# ---------------------------------------------------------------------------
# Small logger that doubles as a wandb shim
# ---------------------------------------------------------------------------


class Logger:
    """Wraps wandb. If disabled, prints metrics to stdout."""

    def __init__(self, enabled: bool, project: str, run_name: str, config: dict):
        self.enabled = enabled and _WANDB_OK
        self.run = None
        if self.enabled:
            import wandb
            self.run = wandb.init(project=project, name=run_name, config=config)
        else:
            if enabled and not _WANDB_OK:
                print("[logger] wandb not installed; falling back to stdout")

    def log(self, data: dict, step: int | None = None):
        if self.enabled:
            import wandb
            payload = {}
            for k, v in data.items():
                payload[k] = v
            wandb.log(payload, step=step)
        else:
            scalars = {
                k: (float(v) if isinstance(v, (int, float, torch.Tensor)) else type(v).__name__)
                for k, v in data.items()
            }
            print(f"[step {step}] " + " ".join(f"{k}={v}" for k, v in scalars.items()))

    def log_image(self, key: str, pil_image: Image.Image, step: int, caption: str = ""):
        if self.enabled:
            import wandb
            wandb.log({key: wandb.Image(pil_image, caption=caption)}, step=step)
        else:
            out = Path(CFG.CACHE_DIR) / f"val_step{step:05d}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            pil_image.save(out)
            print(f"[step {step}] saved {key} -> {out} ({caption})")

    def log_video(self, key: str, frames_uint8: np.ndarray, fps: int, step: int | None = None):
        # frames_uint8: (T, H, W, C) uint8
        if self.enabled:
            import wandb
            # wandb.Video expects (T, C, H, W)
            arr = np.transpose(frames_uint8, (0, 3, 1, 2))
            wandb.log({key: wandb.Video(arr, fps=fps, format="mp4")}, step=step)

    def finish(self):
        if self.enabled:
            import wandb
            wandb.finish()


# ---------------------------------------------------------------------------
# 3. download_and_subset_dataset
# ---------------------------------------------------------------------------


def download_and_subset_dataset(cfg: Config) -> list[tuple[Image.Image, str]]:
    """Pull metadata.parquet + one images zip directly from the HF hub.

    The dataset still ships a `diffusiondb-pixelart.py` loader script, but
    `datasets >= 4` removed loader-script support, so we bypass `load_dataset`
    and read the underlying parquet + zip ourselves. Same end result.
    """
    print(f"[data] downloading metadata.parquet from {cfg.DATASET_ID} ...")
    meta_path = hf_hub_download(
        repo_id=cfg.DATASET_ID, filename="metadata.parquet", repo_type="dataset"
    )
    df = pd.read_parquet(meta_path)

    # Dataset card schema: image_name, prompt (older variants call it "text"), part_id, ...
    text_col = "prompt" if "prompt" in df.columns else "text"
    if text_col not in df.columns:
        raise RuntimeError(f"no prompt/text column in metadata; have {list(df.columns)}")
    if "image_name" not in df.columns or "part_id" not in df.columns:
        raise RuntimeError(f"unexpected metadata schema; have {list(df.columns)}")

    # This repo inherits a 2M-row metadata table from poloclub/diffusiondb but
    # only actually ships images/part-000001.zip and images/part-000002.zip.
    # Discover which zips exist and restrict to those before shuffling.
    from huggingface_hub import HfApi
    repo_files = HfApi().list_repo_files(cfg.DATASET_ID, repo_type="dataset")
    available_parts = set()
    for f in repo_files:
        if f.startswith("images/part-") and f.endswith(".zip"):
            try:
                available_parts.add(int(Path(f).stem.split("-")[-1]))
            except ValueError:
                pass
    if not available_parts:
        raise RuntimeError(f"no images/part-*.zip files in {cfg.DATASET_ID}")
    print(f"[data] available image parts: {sorted(available_parts)} "
          f"(out of {df['part_id'].nunique()} referenced in metadata)")
    df = df[df["part_id"].isin(available_parts)].reset_index(drop=True)
    df = df[df[text_col].astype(str).str.strip() != ""].reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)

    samples: list[tuple[Image.Image, str]] = []
    open_zips: dict[int, zipfile.ZipFile] = {}
    try:
        for _, row in df.iterrows():
            if len(samples) >= cfg.SUBSET_SIZE:
                break
            part_id = int(row["part_id"])
            if part_id not in open_zips:
                # part_id is 1-based and zero-padded to 6 digits in the zip name.
                zip_name = f"images/part-{part_id:06d}.zip"
                print(f"[data] downloading {zip_name} (one-time, ~1GB) ...")
                try:
                    zip_path = hf_hub_download(
                        repo_id=cfg.DATASET_ID, filename=zip_name, repo_type="dataset"
                    )
                except Exception as e:
                    print(f"[data] skipping part {part_id}: {e}")
                    continue
                open_zips[part_id] = zipfile.ZipFile(zip_path, "r")
            zf = open_zips[part_id]
            img_name = str(row["image_name"])
            try:
                with zf.open(img_name) as fp:
                    img = Image.open(io.BytesIO(fp.read())).convert("RGB")
            except KeyError:
                continue  # image listed in metadata but absent from this zip
            samples.append((img, str(row[text_col]).strip()))
    finally:
        for zf in open_zips.values():
            zf.close()

    print(f"[data] kept {len(samples)} samples (asked for {cfg.SUBSET_SIZE}; "
          f"opened {len(open_zips)} zip(s))")
    if not samples:
        raise RuntimeError("no usable samples retrieved from dataset")
    return samples


# ---------------------------------------------------------------------------
# 4. precompute_latents_and_embeds
# ---------------------------------------------------------------------------


def _center_crop_resize(img: Image.Image, h: int, w: int) -> Image.Image:
    iw, ih = img.size
    target_ratio = w / h
    src_ratio = iw / ih
    if src_ratio > target_ratio:
        # Crop horizontally
        new_w = int(round(ih * target_ratio))
        x0 = (iw - new_w) // 2
        img = img.crop((x0, 0, x0 + new_w, ih))
    else:
        new_h = int(round(iw / target_ratio))
        y0 = (ih - new_h) // 2
        img = img.crop((0, y0, iw, y0 + new_h))
    return img.resize((w, h), Image.LANCZOS)


def _pil_to_video_tensor(img: Image.Image, h: int, w: int) -> torch.Tensor:
    """PIL RGB -> tensor shape (1, 3, 1, H, W) in [-1, 1] on CPU."""
    img = _center_crop_resize(img, h, w)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = arr * 2.0 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)
    return t.unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)


def _wan_latents_normalize(latents: torch.Tensor, vae: AutoencoderKLWan) -> torch.Tensor:
    """Match the per-channel mean/std normalization WanPipeline applies."""
    mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype, device=latents.device)
    std = torch.tensor(vae.config.latents_std, dtype=latents.dtype, device=latents.device)
    # latents shape (B, C, F, H, W); mean/std are length-C lists
    mean = mean.view(1, -1, 1, 1, 1)
    std = std.view(1, -1, 1, 1, 1)
    return (latents - mean) * (1.0 / std)


def precompute_latents_and_embeds(cfg: Config, samples: list[tuple[Image.Image, str]] | None = None):
    cache = Path(cfg.CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    samples_dir = cache / "samples"
    samples_dir.mkdir(exist_ok=True)

    if samples is None:
        samples = download_and_subset_dataset(cfg)

    dev = _devmgr()

    # --- VAE encoder on TT (sharded + compiled wrapper) ---
    vae_dtype = cfg.VAE_PRECOMPUTE_DTYPE
    print(f"[precompute] loading VAE from {cfg.MODEL_ID} (dtype={vae_dtype}) ...")
    vae = AutoencoderKLWan.from_pretrained(
        cfg.MODEL_ID, subfolder="vae", torch_dtype=vae_dtype, low_cpu_mem_usage=True,
    ).eval()
    vae = dev.to_device(vae)
    dev.shard_module(vae, "vae_encoder")
    vae_enc_wrapper = _VAEEncoderWrapper(vae)
    vae_enc_compiled = dev.compile(vae_enc_wrapper)

    metadata: list[dict] = []
    print(f"[precompute] VAE-encoding {len(samples)} images at {cfg.TRAIN_H}x{cfg.TRAIN_W} ...")
    with torch.no_grad():
        for i, (img, caption) in enumerate(samples):
            # _pil_to_video_tensor stays on CPU (pure preprocessing); move
            # the (1,3,1,H,W) tensor to TT for the VAE encode.
            video_cpu = _pil_to_video_tensor(img, cfg.TRAIN_H, cfg.TRAIN_W).to(vae_dtype)
            video = dev.to_device(video_cpu)
            latent = vae_enc_compiled(video)
            latent = _wan_latents_normalize(latent, vae)
            latent = latent.squeeze(0).contiguous().to("cpu")  # (C, F, H, W)
            triggered = cfg.TRIGGER + caption
            torch.save(
                {"latent": latent, "caption": triggered},
                samples_dir / f"sample_{i:04d}.pt",
            )
            metadata.append({"idx": i, "caption": triggered})
            if (i + 1) % 8 == 0 or i == len(samples) - 1:
                print(f"  [precompute] {i + 1}/{len(samples)} latent.shape={tuple(latent.shape)}")

    with open(cache / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    del vae_enc_compiled, vae_enc_wrapper, vae
    gc.collect()

    # --- UMT5 text encoder on TT (sharded + compiled wrapper) ---
    print(f"[precompute] loading UMT5 text encoder from {cfg.MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        cfg.MODEL_ID, subfolder="text_encoder", torch_dtype=cfg.DTYPE, low_cpu_mem_usage=True,
    ).eval()
    _truncate_umt5_layers(text_encoder, cfg.DEBUG_UMT5_LAYERS)
    # The checkpoint stores only `shared.weight` and relies on weight-tying for
    # `encoder.embed_tokens.weight`. Our pinned transformers version does not
    # auto-tie on this subfolder load, so without this every forward returns 0.
    # See tt-xla:ppadjin/wan5b_tests/tests/torch/models/wan5b/shared.py:load_umt5.
    text_encoder.encoder.embed_tokens.weight = text_encoder.shared.weight
    text_encoder = dev.to_device(text_encoder)
    dev.shard_module(text_encoder, "umt5")
    umt5_wrapper = _UMT5Wrapper(text_encoder)
    umt5_compiled = dev.compile(umt5_wrapper)

    unique_captions: list[str] = sorted({m["caption"] for m in metadata})
    if "" not in unique_captions:
        unique_captions.append("")
    embeds: dict[str, torch.Tensor] = {}
    print(f"[precompute] T5-encoding {len(unique_captions)} unique captions ...")
    max_seq = 512
    with torch.no_grad():
        for i, cap in enumerate(unique_captions):
            tok = tokenizer(
                cap,
                padding="max_length",
                truncation=True,
                max_length=max_seq,
                return_tensors="pt",
            )
            input_ids = dev.to_device(tok.input_ids)
            attn_mask = dev.to_device(tok.attention_mask)
            seq_lens = tok.attention_mask.sum(dim=1).tolist()
            out = umt5_compiled(input_ids, attn_mask)
            # Match WanPipeline.encode_prompt: zero-out padding then keep full length.
            out = out * attn_mask.unsqueeze(-1).to(out.dtype)
            embeds[cap] = out.squeeze(0).to("cpu")
            if (i + 1) % 10 == 0 or i == len(unique_captions) - 1:
                print(f"  [precompute] {i + 1}/{len(unique_captions)} seq_len={seq_lens[0]}")

    torch.save(embeds, cache / "embeds.pt")
    del umt5_compiled, umt5_wrapper, text_encoder, tokenizer
    gc.collect()
    print(f"[precompute] done. cache at {cache.resolve()}")


# ---------------------------------------------------------------------------
# 5. Dataset + collate
# ---------------------------------------------------------------------------


class PixelArtLatentDataset(Dataset):
    def __init__(self, cache_dir: str, indices: list[int]):
        self.cache_dir = Path(cache_dir)
        self.samples_dir = self.cache_dir / "samples"
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        data = torch.load(self.samples_dir / f"sample_{idx:04d}.pt", weights_only=False)
        return {"latent": data["latent"], "caption": data["caption"], "idx": idx}


def make_collate_fn(embeds: dict[str, torch.Tensor], p_drop: float, seed: int = 0):
    rng = random.Random(seed)

    def collate(batch: list[dict]) -> dict:
        latents = []
        text_embeds = []
        captions_used: list[str] = []
        idxs: list[int] = []
        for item in batch:
            cap = item["caption"]
            if rng.random() < p_drop:
                cap = ""
            if cap not in embeds:
                raise KeyError(f"missing precomputed embed for caption {cap!r}")
            latents.append(item["latent"])
            text_embeds.append(embeds[cap])
            captions_used.append(cap)
            idxs.append(item["idx"])
        return {
            "latent": torch.stack(latents, dim=0),
            "text_embed": torch.stack(text_embeds, dim=0),
            "captions": captions_used,
            "idx": idxs,
        }

    return collate


# ---------------------------------------------------------------------------
# 6. LoRA transformer
# ---------------------------------------------------------------------------


def _truncate_umt5_layers(encoder: UMT5EncoderModel, n: int) -> None:
    """Keep only the first `n` UMT5 encoder layers. `n<=0` -> full model."""
    if n and n > 0:
        encoder.encoder.block = nn.ModuleList(list(encoder.encoder.block[:n]))
        print(f"[debug] truncated UMT5 to {n} layer(s)")


def _truncate_dit_blocks(transformer: WanTransformer3DModel, n: int) -> None:
    """Keep only the first `n` DiT blocks. `n<=0` -> full model."""
    if n and n > 0:
        transformer.blocks = nn.ModuleList(list(transformer.blocks[:n]))
        print(f"[debug] truncated DiT to {n} block(s)")


def build_lora_transformer(cfg: Config) -> WanTransformer3DModel:
    print(f"[lora] loading transformer from {cfg.MODEL_ID} (dtype={cfg.DTYPE}) ...")
    transformer = WanTransformer3DModel.from_pretrained(
        cfg.MODEL_ID, subfolder="transformer", torch_dtype=cfg.DTYPE, low_cpu_mem_usage=True,
    )
    _truncate_dit_blocks(transformer, cfg.DEBUG_DIT_BLOCKS)
    dev = _devmgr()
    transformer = dev.to_device(transformer)
    for p in transformer.parameters():
        p.requires_grad_(False)
    if cfg.GRADIENT_CHECKPOINTING and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
        print("[lora] gradient checkpointing enabled")
    lora_cfg = LoraConfig(
        r=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        target_modules=list(cfg.LORA_TARGETS),
        lora_dropout=0.0,
        init_lora_weights="gaussian",
    )
    transformer.add_adapter(lora_cfg)
    dev.shard_module(transformer, "dit")
    total = sum(p.numel() for p in transformer.parameters())
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"[lora] total={total / 1e6:.1f}M trainable={trainable / 1e6:.2f}M "
          f"(rank={cfg.LORA_RANK}, alpha={cfg.LORA_ALPHA}, targets={list(cfg.LORA_TARGETS)})")
    assert trainable > 0, "no trainable LoRA params; check target_modules"
    assert trainable < total // 20, "trainable params suspiciously large; LoRA not isolated"
    return transformer


# ---------------------------------------------------------------------------
# 7. Flow-matching loss step
# ---------------------------------------------------------------------------


def _sample_timesteps(batch_size: int, cfg: Config, generator: torch.Generator | None = None) -> torch.Tensor:
    """SD3-style logit-normal + flow shift. Returns t in [0,1] of shape (B,)
    on CPU (caller moves to TT before use)."""
    if generator is None:
        u = torch.randn(batch_size) * cfg.LOGNORM_STD + cfg.LOGNORM_MEAN
    else:
        u = torch.randn(batch_size, generator=generator) * cfg.LOGNORM_STD + cfg.LOGNORM_MEAN
    u = torch.sigmoid(u)
    shift = cfg.TRAIN_FLOW_SHIFT
    t = shift * u / (1.0 + (shift - 1.0) * u)
    return t


def flow_matching_step(
    transformer: WanTransformer3DModel,
    batch: dict,
    cfg: Config,
    *,
    fixed_t: torch.Tensor | None = None,
    fixed_noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (loss, t) where t is the per-sample σ in [0,1] of shape (B,).

    Noise + timesteps are always generated on CPU and transferred to TT.
    """
    dev = _devmgr()

    x0 = batch["latent"].to(cfg.DTYPE)
    text_embed = batch["text_embed"].to(cfg.DTYPE)
    B = x0.shape[0]

    if fixed_t is None:
        t = _sample_timesteps(B, cfg)        # CPU
    else:
        t = fixed_t                          # already CPU per caller
    if fixed_noise is None:
        noise = torch.randn(x0.shape, dtype=x0.dtype)   # CPU
    else:
        noise = fixed_noise.to(x0.dtype)

    # CPU -> TT transfers.
    x0 = dev.to_device(x0)
    text_embed = dev.to_device(text_embed)
    t = dev.to_device(t.to(cfg.DTYPE))
    noise = dev.to_device(noise)

    sigma = t.view(B, 1, 1, 1, 1)
    timestep = (t * 1000.0).long()

    x_t = (1.0 - sigma) * x0 + sigma * noise
    pred = transformer(
        hidden_states=x_t,
        timestep=timestep,
        encoder_hidden_states=text_embed,
        return_dict=True,
    ).sample
    target = noise - x0
    loss = F.mse_loss(pred.float(), target.float())
    return loss, t.detach().to("cpu")


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().to("cpu", torch.float32).flatten()
    b = b.detach().to("cpu", torch.float32).flatten()
    va, vb = a - a.mean(), b - b.mean()
    denom = va.norm() * vb.norm()
    return float("nan") if denom == 0 else float((va @ vb) / denom)


def _lora_grad_vec(model: nn.Module) -> dict[str, torch.Tensor]:
    """Flattened LoRA grads keyed by param name (CPU float32)."""
    out = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and "lora_" in name:
            out[name] = p.grad.detach().to("cpu", torch.float32).flatten()
    return out


def _build_cpu_replica(cfg: Config, tt_transformer: WanTransformer3DModel) -> WanTransformer3DModel:
    """Same arch + same (already-LoRA'd) weights as the TT transformer, on CPU."""
    m = WanTransformer3DModel.from_pretrained(
        cfg.MODEL_ID, subfolder="transformer", torch_dtype=cfg.DTYPE, low_cpu_mem_usage=True,
    )
    _truncate_dit_blocks(m, cfg.DEBUG_DIT_BLOCKS)
    for p in m.parameters():
        p.requires_grad_(False)
    lora_cfg = LoraConfig(
        r=cfg.LORA_RANK, lora_alpha=cfg.LORA_ALPHA,
        target_modules=list(cfg.LORA_TARGETS), lora_dropout=0.0,
        init_lora_weights="gaussian",
    )
    m.add_adapter(lora_cfg)
    sd = {k: v.detach().to("cpu", cfg.DTYPE) for k, v in tt_transformer.state_dict().items()}
    m.load_state_dict(sd, strict=True)
    return m.to("cpu").eval()


def _load_debug_dit(cfg: Config) -> WanTransformer3DModel:
    """Fresh truncated DiT for debug probes — no LoRA, eval, frozen."""
    m = WanTransformer3DModel.from_pretrained(
        cfg.MODEL_ID,
        subfolder="transformer",
        torch_dtype=cfg.DTYPE,
        low_cpu_mem_usage=True,
    )
    _truncate_dit_blocks(m, cfg.DEBUG_DIT_BLOCKS)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def _debug_first_step_tt_vs_cpu(cfg, transformer, compiled_transformer, batch):
    """Compare dev_wan-style TT compile vs CPU on one forward (debug only)."""
    import torch_xla

    def _dit_per_patch_timestep(
        t: torch.Tensor,
        latent: torch.Tensor,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> torch.Tensor:
        """dev_wan / expand_timesteps: (B, num_patches) float32 (debug only)."""
        B = latent.shape[0]
        _, _, fl, hl, wl = latent.shape
        pt, ph, pw = patch_size
        seq = (fl // pt) * (hl // ph) * (wl // pw)
        return (t.to(torch.float32) * 1000.0).view(B, 1).expand(B, seq).contiguous()

    dev = _devmgr()
    B = batch["latent"].shape[0]
    x0 = batch["latent"].to(cfg.DTYPE)
    text = batch["text_embed"].to(cfg.DTYPE)
    t = _sample_timesteps(B, cfg).to(cfg.DTYPE)
    noise = torch.randn(x0.shape, dtype=x0.dtype)
    sigma = t.view(B, 1, 1, 1, 1)
    x_t = (1.0 - sigma) * x0 + sigma * noise
    pt, ph, pw = (1, 2, 2)
    timestep = _dit_per_patch_timestep(t, x0, patch_size=(pt, ph, pw))
    target = (noise - x0).float()

    print(
        f"[debug] dev_wan-style probe  blocks={cfg.DEBUG_DIT_BLOCKS}  "
        f"fresh compile after .to(device), enable_trace=True, no LoRA  "
        f"x_t={tuple(x_t.shape)}  timestep={tuple(timestep.shape)} {timestep.dtype}  "
        f"target={tuple(target.shape)}",
        flush=True,
    )

    cpu_wrapper = _WanDiTWrapper(_load_debug_dit(cfg)).to(cfg.DTYPE)

    tt_wrapper = _WanDiTWrapper(_load_debug_dit(cfg)).to(cfg.DTYPE)
    tt_wrapper = tt_wrapper.to(dev.device)

    debug_xla_opts = {
        "optimization_level": "0",
        "experimental-enable-dram-space-saving-optimization": "true",
        #"enable_trace": "true",
    }
    torch_xla.set_custom_compile_options(debug_xla_opts)
    print(f"[debug] xla compile opts (probe only): {debug_xla_opts}", flush=True)

    compiled_tt = torch.compile(tt_wrapper, backend="tt")
    with torch.no_grad():
        pred_cpu = cpu_wrapper(
            x_t, timestep, text,
        ).float()
        loss_cpu = F.mse_loss(pred_cpu, target)

        pred_tt = compiled_tt(
            dev.to_device(x_t),
            dev.to_device(timestep),
            dev.to_device(text),
        )
        #dev.sync()
        torch_xla.sync(wait=True)
        pred_tt = pred_tt.detach().to("cpu").float()
        loss_tt = F.mse_loss(pred_tt, target)

        print(
            f"[debug] FORWARD  loss_tt={loss_tt.item():.4f}  loss_cpu={loss_cpu.item():.4f}  "
            f"PCC(pred_tt,pred_cpu)={_pcc(pred_tt, pred_cpu):.4f}  "
            f"PCC(pred_tt,target)={_pcc(pred_tt, target):.4f}  "
            f"PCC(pred_cpu,target)={_pcc(pred_cpu, target):.4f}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# 8 + 9. Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validation_loss(
    transformer: WanTransformer3DModel,
    val_loader: DataLoader,
    cfg: Config,
) -> float:
    transformer.eval()
    losses = []
    for batch in val_loader:
        # Per-sample deterministic noise/timestep keyed by the original cache index.
        idx = int(batch["idx"][0])
        g = torch.Generator().manual_seed(cfg.SEED + idx)
        t = _sample_timesteps(batch["latent"].shape[0], cfg, generator=g)
        noise = torch.randn(batch["latent"].shape, generator=g)
        loss, _ = flow_matching_step(transformer, batch, cfg, fixed_t=t, fixed_noise=noise)
        losses.append(loss.item())
    transformer.train()
    return float(np.mean(losses)) if losses else float("nan")


def _build_pipeline_for_validation(transformer: WanTransformer3DModel, cfg: Config) -> WanPipeline:
    """Build a WanPipeline with all components on TT (text_encoder + vae +
    transformer all sharded; transformer reused from the train loop so it
    keeps its compiled cache entry).
    """
    dev = _devmgr()

    vae = AutoencoderKLWan.from_pretrained(
        cfg.MODEL_ID, subfolder="vae", torch_dtype=cfg.DTYPE, low_cpu_mem_usage=True,
    )
    pipe = WanPipeline.from_pretrained(
        cfg.MODEL_ID,
        transformer=transformer,
        vae=vae,
        torch_dtype=cfg.DTYPE,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=cfg.INFER_FLOW_SHIFT
    )

    # Move every model the pipeline uses to TT + shard. text_encoder gets the
    # same `shared.weight` tie as in precompute. transformer is already on TT
    # and sharded (build_lora_transformer did that). VAE encoder shard specs
    # also seed the decoder path because both share `vae.{quant,post_quant}_conv`.
    if getattr(pipe, "text_encoder", None) is not None:
        _truncate_umt5_layers(pipe.text_encoder, cfg.DEBUG_UMT5_LAYERS)
        pipe.text_encoder.encoder.embed_tokens.weight = pipe.text_encoder.shared.weight
        pipe.text_encoder = dev.to_device(pipe.text_encoder)
        dev.shard_module(pipe.text_encoder, "umt5")
    pipe.vae = dev.to_device(pipe.vae)
    dev.shard_module(pipe.vae, "vae_decoder")
    # transformer is already on TT — pipe.transformer is the same object.

    return pipe


def _cache_ctx(model: nn.Module, name: str):
    """`model.cache_context(name)` if the module exposes it (CacheMixin),
    else a no-op. Used to mirror diffusers' per-pass cache bucketing."""
    cc = getattr(model, "cache_context", None)
    if callable(cc):
        try:
            return cc(name)
        except Exception:
            return nullcontext()
    return nullcontext()


@torch.no_grad()
def _generate_wan_video(
    pipe: WanPipeline,
    compiled_transformer,
    cfg: Config,
    *,
    prompt: str,
    negative_prompt: str | None,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    generator: torch.Generator,
    output_type: str = "pil",
    max_sequence_length: int = 512,
):
    """ppadjin-style manual denoise loop (mirror of `WanT2VPipeline.generate`
    in tt-xla:ppadjin/wan5b_tests).

    Why not `pipe(...)`: diffusers' `WanPipeline.__call__` runs the
    *uncompiled* DiT eagerly under lazy tensors and only calls
    `xm.mark_step()` at the end of each loop iteration, so every per-step
    graph fuses the DiT forward **with** the UniPC `scheduler.step`. UniPC's
    per-step sigma ratios + multistep-order branching differ each step, so
    the HLO hash changes every iteration -> PJRT cache miss -> a full
    recompile per step (x2 with CFG).

    Here the *compiled* DiT is the only thing on TT inside the loop, and it
    sees plain device tensors (latent + per-step timestep) that change in
    value but never in shape/dtype. The scheduler step, CFG blend, and all
    per-step scalar math stay on CPU. The DiT HLO is then byte-identical
    across steps: compiled once, every later step is a cache hit.
    """
    dev = _devmgr()
    device = dev.device
    transformer_dtype = cfg.DTYPE
    do_cfg = guidance_scale > 1.0

    vae_t = pipe.vae.config.scale_factor_temporal
    vae_s = pipe.vae.config.scale_factor_spatial
    patch_size = pipe.transformer.config.patch_size

    # --- shape alignment (mirror WanPipeline.__call__) ---
    if num_frames % vae_t != 1:
        num_frames = num_frames // vae_t * vae_t + 1
    num_frames = max(num_frames, 1)
    height = height // (vae_s * patch_size[1]) * (vae_s * patch_size[1])
    width = width // (vae_s * patch_size[2]) * (vae_s * patch_size[2])

    tt_cast = lambda x: x.to(dtype=transformer_dtype, device=device)
    cpu_cast = lambda x: x.to(device="cpu", dtype=torch.float32)

    # --- text encode (UMT5 on TT; torch.compile'd as its own graph, like
    # dev_wan/ppadjin, and identical to the `precompute` path). Compiling the
    # encoder (vs the eager `pipe.encode_prompt`) keeps UMT5 a *separate*
    # cached graph instead of fusing its forward (incl. the 256384x4096 vocab
    # gather) into the first DiT graph. The wrapper is stashed on `pipe` so its
    # id() stays stable across calls (the compile cache is id-keyed).
    umt5 = getattr(pipe, "_compiled_umt5", None)
    if umt5 is None:
        pipe._umt5_wrapper = _UMT5Wrapper(pipe.text_encoder)
        umt5 = dev.compile(pipe._umt5_wrapper)
        pipe._compiled_umt5 = umt5

    def _encode(text: str) -> torch.Tensor:
        tok = pipe.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
        )
        input_ids = dev.to_device(tok.input_ids)
        attn_mask = dev.to_device(tok.attention_mask)
        out = umt5(input_ids, attn_mask)
        # Match WanPipeline.encode_prompt / precompute: zero padding, keep full length.
        out = out * attn_mask.unsqueeze(-1).to(out.dtype)
        return out.to(transformer_dtype)

    prompt_embeds = _encode(prompt)
    negative_prompt_embeds = _encode(negative_prompt or "") if do_cfg else None
    # Cut the graph here so the realized text embeds enter the DiT loop as plain
    # device-buffer inputs (UMT5 stays its own compiled/cached graph).
    xm.mark_step()

    # --- timesteps + latents (CPU) ---
    pipe.scheduler.set_timesteps(num_inference_steps, device="cpu")
    timesteps = pipe.scheduler.timesteps

    num_latent_frames = (num_frames - 1) // vae_t + 1
    latent_shape = (
        1,
        pipe.transformer.config.in_channels,
        num_latent_frames,
        height // vae_s,
        width // vae_s,
    )
    latents = torch.randn(latent_shape, generator=generator, dtype=torch.float32, device="cpu")
    mask = torch.ones_like(latents)
    expand_ts = bool(getattr(pipe.config, "expand_timesteps", False))

    for t in timesteps:
        latent_model_input = latents.to(transformer_dtype)        # CPU
        if expand_ts:
            # per-patch timestep: num_latent_frames * (H//2) * (W//2)
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(1, -1)
        else:
            timestep = t.expand(1)

        # CPU -> TT: only the DiT inputs cross to the device. timestep keeps
        # its dtype (int64/float) so the compiled graph's input sig is stable.
        lat_dev = tt_cast(latent_model_input)
        ts_dev = timestep.to(device)

        with _cache_ctx(pipe.transformer, "cond"):
            noise_pred = compiled_transformer(
                hidden_states=lat_dev,
                timestep=ts_dev,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
        noise_pred = cpu_cast(noise_pred)        # sync -> per-step graph boundary

        if do_cfg:
            with _cache_ctx(pipe.transformer, "uncond"):
                noise_uncond = compiled_transformer(
                    hidden_states=lat_dev,
                    timestep=ts_dev,
                    encoder_hidden_states=negative_prompt_embeds,
                    return_dict=False,
                )[0]
            noise_uncond = cpu_cast(noise_uncond)
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        # scheduler step on CPU keeps UniPC's per-step scalars out of the DiT graph.
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    if output_type == "latent":
        return latents

    # --- VAE decode (TT, one shot; needs the slice clamp) ---
    latents_vae = latents.to(torch.float32)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents_vae.device, latents_vae.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(latents_vae.device, latents_vae.dtype)
    latents_vae = (latents_vae / latents_std + latents_mean).to(dtype=pipe.vae.dtype, device=device)

    # Compile the whole decode (frame loop + feat_cache) into one graph. The
    # wrapper is stashed on `pipe` so its id() stays stable across calls (the
    # compile cache is id-keyed; a GC'd-then-recreated wrapper could alias).
    vae_decode = getattr(pipe, "_compiled_vae_decode", None)
    if vae_decode is None:
        pipe._vae_dec_wrapper = _VAEDecoderWrapper(pipe.vae)
        vae_decode = dev.compile(pipe._vae_dec_wrapper)
        pipe._compiled_vae_decode = vae_decode

    with safe_xla_slicing():
        video = vae_decode(latents_vae)
    video = video.to("cpu").to(torch.float32)
    return pipe.video_processor.postprocess_video(video, output_type=output_type)


@torch.no_grad()
def generate_validation_sample(
    transformer: WanTransformer3DModel,
    cfg: Config,
    step: int,
) -> tuple[Image.Image, np.ndarray | None]:
    """Run a short val video through the same pipeline final inference uses.

    Returns (first_frame_PIL, all_frames_uint8_np_or_None). If VAL_IMG_FRAMES==1,
    only the still image is meaningful; otherwise the full video can be logged
    to wandb too.

    The generator is on CPU (per the spec for this script): diffusers will
    generate the initial noise on the generator's device and then move it to
    the pipeline's device before the denoise loop.
    """
    print(f"[val-img] step {step}: generating {cfg.VAL_IMG_FRAMES}-frame sample "
          f"({cfg.INFER_H}x{cfg.INFER_W}, {cfg.VAL_IMG_STEPS} steps) ...")
    transformer.eval()
    pipe = _build_pipeline_for_validation(transformer, cfg)
    # Reuse the cached compiled DiT (id-keyed in WanDeviceManager.compile) so
    # the denoise loop hits one compiled graph instead of recompiling per step.
    compiled_transformer = _devmgr().compile(transformer)
    gen = torch.Generator(device="cpu").manual_seed(cfg.SEED)
    t0 = time.time()
    # Manual ppadjin-style loop: compiled DiT on TT, scheduler/CFG/decode-prep
    # on CPU. `safe_xla_slicing` wraps the VAE decode inside the helper.
    video = _generate_wan_video(
        pipe,
        compiled_transformer,
        cfg,
        prompt=cfg.TRIGGER + cfg.VAL_PROMPT,
        negative_prompt=cfg.NEG_PROMPT or None,
        height=cfg.INFER_H,
        width=cfg.INFER_W,
        num_frames=cfg.VAL_IMG_FRAMES,
        num_inference_steps=cfg.VAL_IMG_STEPS,
        guidance_scale=cfg.INFER_GUIDANCE,
        generator=gen,
        output_type="pil",
    )
    frames = video[0]  # list of PIL.Image
    img = frames[0]
    video_np = None
    if len(frames) > 1:
        video_np = np.stack([np.asarray(f) for f in frames], axis=0).astype(np.uint8)
    print(f"[val-img] step {step}: generated in {time.time() - t0:.1f}s, frames={len(frames)}")
    del pipe.vae
    del pipe
    gc.collect()
    transformer.train()
    return img, video_np


# ---------------------------------------------------------------------------
# 10. Train
# ---------------------------------------------------------------------------


def train(cfg: Config):
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    cache = Path(cfg.CACHE_DIR)
    samples_dir = cache / "samples"
    embeds_path = cache / "embeds.pt"
    if not embeds_path.exists() or not samples_dir.exists():
        raise FileNotFoundError(
            f"missing cache at {cache} — run `python wan22_5b.py precompute` first."
        )

    metadata = json.loads((cache / "metadata.json").read_text())
    all_indices = sorted(m["idx"] for m in metadata)
    if len(all_indices) <= cfg.VAL_HOLDOUT:
        raise RuntimeError(f"need > {cfg.VAL_HOLDOUT} cached samples; got {len(all_indices)}")
    val_indices = all_indices[-cfg.VAL_HOLDOUT:]
    train_indices = all_indices[:-cfg.VAL_HOLDOUT]
    print(f"[train] {len(train_indices)} train / {len(val_indices)} val cached samples")

    print(f"[train] loading embeds.pt ({embeds_path}) ...")
    embeds = torch.load(embeds_path, weights_only=False)

    train_ds = PixelArtLatentDataset(cfg.CACHE_DIR, train_indices)
    val_ds = PixelArtLatentDataset(cfg.CACHE_DIR, val_indices)
    train_collate = make_collate_fn(embeds, p_drop=cfg.TEXT_DROP_PROB, seed=cfg.SEED)
    val_collate = make_collate_fn(embeds, p_drop=0.0, seed=cfg.SEED + 1)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH, shuffle=True, collate_fn=train_collate,
        num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=val_collate, num_workers=0,
    )

    transformer = build_lora_transformer(cfg)
    transformer.train()
    # Compile the (already TT + sharded + LoRA'd) transformer. The compiled
    # OptimizedModule is reused for every training step and validation
    # forward — `WanDeviceManager.compile` caches on id(transformer).
    compiled_transformer = _devmgr().compile(transformer)

    trainable = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.999),
        capturable=False,
    )
    # LR scheduler disabled — using constant LR for now. Re-enable by uncommenting:
    # warmup_steps = max(1, int(cfg.MAX_STEPS * cfg.LR_WARMUP_FRAC))
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=cfg.MAX_STEPS,
    # )
    # print(f"[train] lr schedule: cosine w/ {warmup_steps} warmup steps "
    #       f"-> {cfg.MAX_STEPS - warmup_steps} cosine-decay steps (peak lr={cfg.LR})")
    print(f"[train] lr schedule: constant lr={cfg.LR} (no scheduler)")

    run_name = f"wan22_5b_pxa_r{cfg.LORA_RANK}a{cfg.LORA_ALPHA}_{int(time.time() % 1_000_000)}"
    logger = Logger(cfg.WANDB_ENABLED, cfg.WANDB_PROJECT, run_name, cfg.asdict())

    # ----- Pre-training baseline (step=0): skipped — fused eager pipe(UMT5+DiT)
    # blows up TTNN IR (~200k ops from RoPE consteval). Training uses compiled
    # DiT + precomputed embeds only; periodic val videogen still runs later.
    # ----- DEBUG: initial e2e videogen commented out for the first-step PCC probe.
    # print(f"[train] generating step=0 baseline (LoRA disabled) ...")
    # transformer.disable_adapters()
    # try:
    #     baseline_pil, baseline_video = generate_validation_sample(transformer, cfg, step=0)
    #     logger.log_image(
    #         "val/sample", baseline_pil, step=0,
    #         caption=f"step=0 BASELINE (no LoRA) prompt={cfg.TRIGGER + cfg.VAL_PROMPT!r}",
    #     )
    #     if baseline_video is not None:
    #         logger.log_video("val/sample_video", baseline_video, fps=cfg.INFER_FPS, step=0)
    # except Exception as e:
    #     print(f"[train] baseline generation failed ({e!r}); skipping")
    # transformer.enable_adapters()
    # print("[train] skipping step=0 baseline videogen (fused UMT5+DiT compile); starting training loop")
    #exit(0)

    global_step = 0
    ema_loss: float | None = None
    ema_alpha = 0.1
    accum_loss = 0.0
    accum_count = 0
    accum_sigmas: list[float] = []
    micro_step = 0
    step_start = time.time()
    data_iter = iter(train_loader)

    print(f"[train] starting loop: max_steps={cfg.MAX_STEPS}, accum={cfg.GRAD_ACCUM}, "
          f"batch={cfg.BATCH}, lr={cfg.LR}, wd={cfg.WEIGHT_DECAY}")
    optimizer.zero_grad(set_to_none=True)

    # ----- DEBUG: run ONE training step on TT and on a CPU replica with the
    # exact same (x_t, timestep, text_embed, target), compare the DiT output
    # PCC, print both losses, then exit.
    _debug_first_step_tt_vs_cpu(cfg, transformer, compiled_transformer, next(data_iter))
    #_debug_first_step_tt_vs_cpu(cfg, transformer, compiled_transformer, next(data_iter))
    exit(0)

    while global_step < cfg.MAX_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        loss, sigmas = flow_matching_step(compiled_transformer, batch, cfg)
        (loss / cfg.GRAD_ACCUM).backward()
        # Force a sync after backward so loss.item() / sigmas.tolist() below
        # don't trigger N implicit small graph compiles per step.
        _devmgr().sync()
        accum_loss += loss.item()
        accum_count += 1
        accum_sigmas.extend(sigmas.flatten().tolist())
        micro_step += 1

        if micro_step % cfg.GRAD_ACCUM == 0:
            #optimizer.step()
            xm.optimizer_step(optimizer, barrier=True)
            # lr_scheduler.step()
            #optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad()
            _devmgr().sync()
            global_step += 1
            avg_loss = accum_loss / accum_count
            ema_loss = avg_loss if ema_loss is None else (1 - ema_alpha) * ema_loss + ema_alpha * avg_loss
            step_time = time.time() - step_start
            step_start = time.time()
            print(
                f"[train] step {global_step}/{cfg.MAX_STEPS} "
                f"loss={avg_loss:.4f} ema={ema_loss:.4f} step_time={step_time:.2f}s",
                flush=True,
            )
            sig_arr = np.asarray(accum_sigmas)
            log_payload = {
                "train/loss": avg_loss,
                "train/loss_ema": ema_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/step_time_s": step_time,
                "train/sigma_mean": float(sig_arr.mean()),
                "train/sigma_min": float(sig_arr.min()),
                "train/sigma_max": float(sig_arr.max()),
                "train/timestep_mean": float(sig_arr.mean() * 1000.0),
            }
            if logger.enabled:
                import wandb
                log_payload["train/sigma_hist"] = wandb.Histogram(sig_arr)
            logger.log(log_payload, step=global_step)
            accum_loss = 0.0
            accum_count = 0
            accum_sigmas = []

            if cfg.VAL_LOSS_EVERY > 0 and global_step % cfg.VAL_LOSS_EVERY == 0:
                vloss = validation_loss(compiled_transformer, val_loader, cfg)
                logger.log({"val/loss": vloss}, step=global_step)
                print(f"[train] step {global_step}: val/loss={vloss:.4f}")

            if global_step % cfg.VAL_IMG_EVERY == 0:
                # Save an in-progress LoRA checkpoint BEFORE the slow video gen
                # so a crash there can never lose training progress.
                ckpt_path = str(Path(cfg.LORA_PATH).with_name(
                    Path(cfg.LORA_PATH).stem + f"_step{global_step:05d}.safetensors"))
                save_lora(transformer, ckpt_path)
                try:
                    pil, vid = generate_validation_sample(transformer, cfg, global_step)
                    logger.log_image(
                        "val/sample", pil, step=global_step,
                        caption=f"step={global_step} prompt={cfg.TRIGGER + cfg.VAL_PROMPT!r}",
                    )
                    if vid is not None:
                        logger.log_video("val/sample_video", vid, fps=cfg.INFER_FPS, step=global_step)
                except Exception as e:
                    print(f"[val-img] step {global_step}: generation failed ({e!r}); "
                          f"training continues, checkpoint at {ckpt_path}")

    save_lora(transformer, cfg.LORA_PATH)
    print(f"[train] done at step {global_step}; saved LoRA to {cfg.LORA_PATH}")

    # Final video generation + log to wandb.
    infer(cfg, cfg.LORA_PATH, cfg.VAL_PROMPT, logger=logger, transformer=transformer)
    logger.finish()


# ---------------------------------------------------------------------------
# 11. Save LoRA
# ---------------------------------------------------------------------------


def save_lora(transformer: WanTransformer3DModel, path: str):
    state = get_peft_model_state_dict(transformer)
    # Prefix with "transformer." so diffusers' WanPipeline.load_lora_weights()
    # routes these weights to pipe.transformer. Without the prefix, the loader
    # silently no-ops and inference produces an un-LoRA'd (baseline) video.
    state_cpu = {f"transformer.{k}": v.detach().to("cpu").contiguous() for k, v in state.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(state_cpu, path)
    print(f"[save] wrote {len(state_cpu)} LoRA tensors -> {path}")


def _load_lora_compat(pipe: WanPipeline, lora_path: str) -> None:
    """Load LoRA, auto-prefixing legacy keys saved without 'transformer.' prefix.

    Older checkpoints from this script were saved as 'blocks.X.attnY...' (raw
    PEFT format), which silently fails to apply via pipe.load_lora_weights().
    Newer checkpoints are saved as 'transformer.blocks.X.attnY...' and load
    natively. This loader handles both.
    """
    from safetensors.torch import load_file
    sd = load_file(lora_path)
    if not any(k.startswith("transformer.") for k in sd):
        print(f"[infer] legacy LoRA format detected; re-prefixing {len(sd)} keys with 'transformer.'")
        sd = {f"transformer.{k}": v for k, v in sd.items()}
    pipe.load_lora_weights(sd)


# ---------------------------------------------------------------------------
# 12. Inference (final 2-3 s video)
# ---------------------------------------------------------------------------


def infer(
    cfg: Config,
    lora_path: str,
    prompt: str,
    logger: Logger | None = None,
    transformer: WanTransformer3DModel | None = None,
    output_path: str | None = None,
    negative_prompt: str | None = None,
):
    print(f"[infer] building WanPipeline (this loads VAE + UMT5 + transformer if needed) ...")
    dev = _devmgr()

    if transformer is None:
        # Loading from scratch path: load fresh pipe, move every component to
        # TT + shard, then apply LoRA via diffusers loader.
        pipe = WanPipeline.from_pretrained(cfg.MODEL_ID, torch_dtype=cfg.DTYPE)
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=cfg.INFER_FLOW_SHIFT
        )
        if getattr(pipe, "text_encoder", None) is not None:
            _truncate_umt5_layers(pipe.text_encoder, cfg.DEBUG_UMT5_LAYERS)
            pipe.text_encoder.encoder.embed_tokens.weight = pipe.text_encoder.shared.weight
            pipe.text_encoder = dev.to_device(pipe.text_encoder)
            dev.shard_module(pipe.text_encoder, "umt5")
        pipe.vae = dev.to_device(pipe.vae)
        dev.shard_module(pipe.vae, "vae_decoder")
        _truncate_dit_blocks(pipe.transformer, cfg.DEBUG_DIT_BLOCKS)
        pipe.transformer = dev.to_device(pipe.transformer)
        _load_lora_compat(pipe, lora_path)
        dev.shard_module(pipe.transformer, "dit")
    else:
        # Reuse the already-LoRA'd transformer from train().
        vae = AutoencoderKLWan.from_pretrained(
            cfg.MODEL_ID, subfolder="vae", torch_dtype=cfg.DTYPE, low_cpu_mem_usage=True,
        )
        pipe = WanPipeline.from_pretrained(
            cfg.MODEL_ID, transformer=transformer, vae=vae, torch_dtype=cfg.DTYPE,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=cfg.INFER_FLOW_SHIFT
        )
        if getattr(pipe, "text_encoder", None) is not None:
            _truncate_umt5_layers(pipe.text_encoder, cfg.DEBUG_UMT5_LAYERS)
            pipe.text_encoder.encoder.embed_tokens.weight = pipe.text_encoder.shared.weight
            pipe.text_encoder = dev.to_device(pipe.text_encoder)
            dev.shard_module(pipe.text_encoder, "umt5")
        pipe.vae = dev.to_device(pipe.vae)
        dev.shard_module(pipe.vae, "vae_decoder")
        # transformer already on TT + sharded.

    # Default to whatever cfg.NEG_PROMPT says (we set it to empty because the
    # official Wan 2.2 neg prompt actively suppresses style/painting/low-quality
    # — i.e. exactly what a pxa LoRA produces). Pass a string to override.
    if negative_prompt is None:
        negative_prompt = cfg.NEG_PROMPT or None

    print(f"[infer] generating {cfg.INFER_FRAMES} frames @ {cfg.INFER_H}x{cfg.INFER_W} "
          f"in {cfg.INFER_STEPS} steps (cfg={cfg.INFER_GUIDANCE}, flow_shift={cfg.INFER_FLOW_SHIFT})")
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(cfg.SEED)
    pipe.transformer.eval()
    # Compiled DiT (id-keyed cache) so the denoise loop reuses one graph.
    compiled_transformer = dev.compile(pipe.transformer)
    # Manual ppadjin-style loop: compiled DiT on TT, scheduler/CFG/decode-prep
    # on CPU. `safe_xla_slicing` wraps the VAE decode inside the helper.
    video = _generate_wan_video(
        pipe,
        compiled_transformer,
        cfg,
        prompt=cfg.TRIGGER + prompt,
        negative_prompt=negative_prompt,
        height=cfg.INFER_H,
        width=cfg.INFER_W,
        num_frames=cfg.INFER_FRAMES,
        num_inference_steps=cfg.INFER_STEPS,
        guidance_scale=cfg.INFER_GUIDANCE,
        generator=gen,
        output_type="pil",
    )
    frames = video[0]  # list of PIL.Image
    print(f"[infer] generated in {(time.time() - t0) / 60.0:.1f} min; frames={len(frames)}")

    out_path = output_path or cfg.INFER_OUTPUT
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, out_path, fps=cfg.INFER_FPS)
    print(f"[infer] saved video -> {out_path}")

    if logger is not None:
        np_frames = np.stack([np.asarray(f) for f in frames], axis=0).astype(np.uint8)
        logger.log_video("val/final_video", np_frames, fps=cfg.INFER_FPS)


# ---------------------------------------------------------------------------
# 13. CLI
# ---------------------------------------------------------------------------


def _apply_cli_overrides(args, cfg: Config) -> Config:
    if getattr(args, "no_wandb", False):
        cfg.WANDB_ENABLED = False
    if getattr(args, "max_steps", None):
        cfg.MAX_STEPS = args.max_steps
    if getattr(args, "subset_size", None):
        cfg.SUBSET_SIZE = args.subset_size
    if getattr(args, "train_res", None):
        cfg.TRAIN_H = cfg.TRAIN_W = args.train_res
    if getattr(args, "infer_res", None):
        cfg.INFER_H = cfg.INFER_W = args.infer_res
    if getattr(args, "infer_frames", None):
        cfg.INFER_FRAMES = args.infer_frames
    if getattr(args, "infer_steps", None):
        cfg.INFER_STEPS = args.infer_steps
    if getattr(args, "val_img_every", None):
        cfg.VAL_IMG_EVERY = args.val_img_every
    if getattr(args, "val_loss_every", None):
        cfg.VAL_LOSS_EVERY = args.val_loss_every
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sp = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    common.add_argument("--max-steps", type=int, default=None)
    common.add_argument("--subset-size", type=int, default=None)
    common.add_argument("--train-res", type=int, default=None, help="square training resolution")
    common.add_argument("--infer-res", type=int, default=None, help="square inference resolution")
    common.add_argument("--infer-frames", type=int, default=None)
    common.add_argument("--infer-steps", type=int, default=None)
    common.add_argument("--val-img-every", type=int, default=None)
    common.add_argument("--val-loss-every", type=int, default=None)

    sp.add_parser("precompute", parents=[common], help="encode subset with VAE + T5; write cache/")
    sp.add_parser("train", parents=[common], help="LoRA fine-tune the DiT")
    p_inf = sp.add_parser("infer", parents=[common], help="generate final video from a LoRA")
    p_inf.add_argument("--prompt", type=str, default=None)
    p_inf.add_argument("--lora", type=str, default=None)
    p_inf.add_argument("--output", type=str, default=None, help="output mp4 path (overrides INFER_OUTPUT)")
    p_inf.add_argument("--negative-prompt", type=str, default=None,
                       help="negative prompt; defaults to the official Wan2.2 one")
    sp.add_parser("all", parents=[common], help="precompute -> train -> infer")

    args = p.parse_args()
    cfg = _apply_cli_overrides(args, CFG)

    # Force the device manager to come up once at the start so XLA SPMD +
    # custom compile options are set before any model load.
    _devmgr()

    if args.cmd == "precompute":
        precompute_latents_and_embeds(cfg)
    elif args.cmd == "train":
        checks_config = chisel.ChiselChecksConfig(
            isolation=True,
            accumulation=False,
        )
        with chisel.session(results_path="output_report.jsonl", checks_config=checks_config) as report:
            train(cfg)
    elif args.cmd == "infer":
        lora_path = args.lora or cfg.LORA_PATH
        prompt = args.prompt or cfg.VAL_PROMPT
        infer(cfg, lora_path, prompt,
              output_path=args.output,
              negative_prompt=args.negative_prompt)
    elif args.cmd == "all":
        precompute_latents_and_embeds(cfg)
        checks_config = chisel.ChiselChecksConfig(
            isolation=True,
            accumulation=False,
        )
        with chisel.session(results_path="output_report.jsonl", checks_config=checks_config) as report:
            train(cfg)
        # train() already calls infer() at the end.


if __name__ == "__main__":
    main()
