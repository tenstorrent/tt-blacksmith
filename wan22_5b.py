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
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict  # noqa: E402
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
    DEBUG_UMT5_LAYERS: int = 0
    DEBUG_DIT_BLOCKS: int = 10

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

    # LoRA. Matches tmp.py exactly: self-attn (attn1.*) + cross-attn (attn2.*)
    # q/k/v/out projections. The "ff.net.*" entries mirror tmp.py but match no
    # module in this model (the block FFN is named "ffn"), so the adapted set is
    # attention-only (480 modules) — byte-identical to the wan22_5b_gpu ckpt.
    LORA_RANK: int = 32
    LORA_ALPHA: int = 32                       # scale = alpha/r = 1.0
    LORA_TARGETS: tuple = (
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.net.0.proj", "ff.net.2",
        # "ffn.net.0.proj", "ffn.net.2",   # FFN LoRA — uncomment to restore attn+FFN
    )
    # Only attach LoRA to DiT blocks with index >= LORA_MIN_BLOCK (0 = all blocks).
    # The QK/softmax-path grads in the early blocks (0..3) have poor TT-vs-CPU PCC;
    # skipping them keeps trainable params on the numerically-clean blocks only.
    LORA_MIN_BLOCK: int = 0

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

    # Resume. If RESUME_PATH points at a LoRA .safetensors, its weights are
    # loaded before the loop; the sibling "<stem>.opt.pt" (if present) restores
    # optimizer state + global_step + RNG for an exact continuation. Empty =
    # start fresh from step 0. RESUME_STEP is only a fallback step counter for
    # weight-only checkpoints that have no .opt.pt.
    RESUME_PATH: str = ""
    RESUME_STEP: int = 0

    def asdict(self) -> dict:
        d = asdict(self)
        d["DTYPE"] = str(self.DTYPE)
        d["LORA_TARGETS"] = list(self.LORA_TARGETS)
        return d

    def __post_init__(self):
        # Precision override for debugging the bf16 noise floor. The TT loss
        # rides ~0.02 above the GPU golden with PCC(pred)~0.9998 — the signature
        # of bf16 rounding noise adding a positive bias to MSE. Setting
        # WAN_DTYPE=fp32 runs the whole DiT forward+backward in fp32 so we can
        # check whether the loss/quality gap collapses to the reference.
        _dtype_env = os.environ.get("WAN_DTYPE", "").lower()
        if _dtype_env in ("fp32", "float32", "f32"):
            self.DTYPE = torch.float32
        elif _dtype_env in ("bf16", "bfloat16"):
            self.DTYPE = torch.bfloat16

        # Quick experiment overrides: number of DiT blocks and the first block
        # that gets a LoRA adapter (skip the numerically-noisy early blocks).
        _blocks_env = os.environ.get("WAN_DIT_BLOCKS", "")
        if _blocks_env.strip():
            self.DEBUG_DIT_BLOCKS = int(_blocks_env)
        _minblk_env = os.environ.get("WAN_LORA_MIN_BLOCK", "")
        if _minblk_env.strip():
            self.LORA_MIN_BLOCK = int(_minblk_env)

        # WAN_OUT_DIR redirects all training artifacts (LoRA checkpoints,
        # optimizer .opt.pt, in-progress ckpts, final video) into a new dir,
        # so a run can't clobber a previous one.
        _out_dir = os.environ.get("WAN_OUT_DIR", "").strip()
        if _out_dir:
            os.makedirs(_out_dir, exist_ok=True)
            self.LORA_PATH = os.path.join(_out_dir, os.path.basename(self.LORA_PATH))
            self.INFER_OUTPUT = os.path.join(_out_dir, os.path.basename(self.INFER_OUTPUT))

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
        # WAN_INFER_DEVICE=cpu|cuda runs the SAME pipeline + manual denoise loop
        # eagerly on a non-TT device, so per-step tensors can be compared to a
        # TT run. Default "tt" keeps the compiled + sharded TT path.
        _dev = os.environ.get("WAN_INFER_DEVICE", "tt").lower()
        if _dev in ("", "tt"):
            DEVMGR = WanDeviceManager(use_tt=True, sharded=True)
        else:
            DEVMGR = WanDeviceManager(use_tt=False, sharded=False)
            DEVMGR.device = torch.device("cuda" if _dev == "cuda" else "cpu")
            print(f"[device] non-TT reference device = {DEVMGR.device}", flush=True)
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


def _make_lora_config(cfg: Config) -> LoraConfig:
    """LoraConfig honoring LORA_MIN_BLOCK: when >0, only blocks with index
    >= LORA_MIN_BLOCK get adapters (PEFT layers_to_transform + layers_pattern).
    `n_blocks` defaults to DEBUG_DIT_BLOCKS, else assume the full 30-block model."""
    kw = dict(
        r=cfg.LORA_RANK,
        lora_alpha=cfg.LORA_ALPHA,
        target_modules=list(cfg.LORA_TARGETS),
        lora_dropout=0.0,
        init_lora_weights="gaussian",
    )
    if cfg.LORA_MIN_BLOCK and cfg.LORA_MIN_BLOCK > 0:
        n_blocks = cfg.DEBUG_DIT_BLOCKS if cfg.DEBUG_DIT_BLOCKS and cfg.DEBUG_DIT_BLOCKS > 0 else 30
        layers = list(range(cfg.LORA_MIN_BLOCK, n_blocks))
        # PEFT's layers_pattern builds regex `.*\.blocks\.(\d+)\.`, which needs a
        # dot BEFORE "blocks"; but the WAN block list is the top-level attribute
        # (keys are "blocks.4.attn1.to_q"), so that match fails. Instead encode
        # the allowed block indices directly into a regex target_modules string.
        idxs = "|".join(str(i) for i in layers)
        suffixes = "|".join(t.replace(".", r"\.") for t in cfg.LORA_TARGETS)
        kw["target_modules"] = rf".*blocks\.(?:{idxs})\..*\.(?:{suffixes})"
        print(f"[lora] restricting adapters to blocks {layers[0]}..{layers[-1]} "
              f"(LORA_MIN_BLOCK={cfg.LORA_MIN_BLOCK}, n_blocks={n_blocks})")
    return LoraConfig(**kw)


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
    lora_cfg = _make_lora_config(cfg)
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
    return_pred: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the scalar loss (or `(loss, pred)` when `return_pred=True`).

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
    if return_pred:
        return loss, pred
    return loss


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


def _lora_param_vec(model: nn.Module) -> dict[str, torch.Tensor]:
    """Flattened LoRA param values keyed by param name (CPU float32)."""
    out = {}
    for name, p in model.named_parameters():
        if p.requires_grad and "lora_" in name:
            out[name] = p.detach().to("cpu", torch.float32).flatten()
    return out


def _print_lora_grad_pcc(cpu_grads: dict[str, torch.Tensor], tt_grads: dict[str, torch.Tensor]) -> None:
    names = sorted(set(cpu_grads) & set(tt_grads))
    if not names:
        print("[debug] BACKWARD  no LoRA grads to compare", flush=True)
        return
    pccs: list[float] = []
    for name in names:
        g_cpu, g_tt = cpu_grads[name], tt_grads[name]
        if g_cpu.numel() != g_tt.numel():
            print(f"[debug] BACKWARD  {name}: shape mismatch {g_cpu.numel()} vs {g_tt.numel()}", flush=True)
            continue
        if g_cpu.norm() == 0 and g_tt.norm() == 0:
            print(f"[debug] BACKWARD  {name}: PCC=nan (both zero)", flush=True)
            continue
        pcc = _pcc(g_cpu, g_tt)
        pccs.append(pcc)
        print(f"[debug] BACKWARD  {name}: PCC={pcc:.4f}", flush=True)
    if pccs:
        print(
            f"[debug] BACKWARD  LoRA grad PCC  min={min(pccs):.4f}  mean={sum(pccs)/len(pccs):.4f}  "
            f"n={len(pccs)}",
            flush=True,
        )


def _register_io_hooks(model: nn.Module, name_filter=None):
    """Forward hooks on every module that capture (a) the module's output
    activation and (b) the gradient flowing into that output. Keyed by the
    fully-qualified module name. Tensors are stored as-is (possibly lazy XLA
    tensors); convert to CPU f32 only after a sync.

    Returns (fwd, grad, handles, order) where `order` is the module list in
    registration (≈forward) order so the report can be read top-to-bottom.
    """
    fwd: dict[str, torch.Tensor] = {}
    grad: dict[str, torch.Tensor] = {}
    handles = []
    order: list[str] = []

    def make_hook(name: str):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if not isinstance(t, torch.Tensor) or not t.is_floating_point():
                return
            fwd[name] = t.detach()
            if t.requires_grad:
                t.register_hook(lambda g, _n=name: grad.__setitem__(_n, g.detach()))

        return hook

    for name, mod in model.named_modules():
        if name == "":
            continue
        if name_filter is not None and not name_filter(name):
            continue
        order.append(name)
        handles.append(mod.register_forward_hook(make_hook(name)))
    return fwd, grad, handles, order


def _compare_layer_pccs(cfg, transformer, cpu_model, *, cpu_in, tt_in):
    """Per-module forward-output and backward-grad PCC between the eager TT
    module and the CPU replica, using identical inputs. Prints one row per
    module (in forward order) with both the output PCC and the grad PCC, so you
    can read where each falls off. Gated by WAN_DEBUG_HOOKS=1.

    NOTE: hooks fire on the EAGER device path, which is less precise than the
    compiled training path (no composite/fusion), so absolute PCCs are
    pessimistic — read the per-module *trend*, not the absolute values.
    """
    import torch_xla

    x_t, timestep, text, target = cpu_in
    x_t_tt, timestep_tt, text_tt, target_tt = tt_in

    # Skip identity/no-op passthroughs that carry no useful signal.
    def keep(name: str) -> bool:
        return "lora_dropout" not in name

    for p in cpu_model.parameters():
        p.grad = None
    for p in transformer.parameters():
        p.grad = None

    # ----- CPU (reference) eager fwd+bwd with hooks -----
    cpu_fwd, cpu_grad, cpu_h, _ = _register_io_hooks(cpu_model, keep)
    pred_cpu = cpu_model(
        hidden_states=x_t, timestep=timestep,
        encoder_hidden_states=text, return_dict=True,
    ).sample.float()
    F.mse_loss(pred_cpu, target.float()).backward()
    for h in cpu_h:
        h.remove()
    cpu_fwd = {k: v.to("cpu", torch.float32) for k, v in cpu_fwd.items()}
    cpu_grad = {k: v.to("cpu", torch.float32) for k, v in cpu_grad.items()}

    # ----- TT eager fwd+bwd with hooks (NOT the compiled path) -----
    tt_fwd, tt_grad, tt_h, order = _register_io_hooks(transformer, keep)
    pred_tt = transformer(
        hidden_states=x_t_tt, timestep=timestep_tt,
        encoder_hidden_states=text_tt, return_dict=True,
    ).sample
    F.mse_loss(pred_tt.float(), target_tt.float()).backward()
    torch_xla.sync(wait=True)
    for h in tt_h:
        h.remove()
    tt_fwd = {k: v.to("cpu", torch.float32) for k, v in tt_fwd.items()}
    tt_grad = {k: v.to("cpu", torch.float32) for k, v in tt_grad.items()}

    def _cell(cpu_d: dict, tt_d: dict, name: str) -> str:
        if name not in cpu_d or name not in tt_d:
            return "   .  "
        a, b = cpu_d[name], tt_d[name]
        if a.numel() != b.numel() or (a.norm() == 0 and b.norm() == 0):
            return "   .  "
        return f"{_pcc(a, b):+.4f}"

    print(f"[hooks] {'module':50s} {'out_pcc':>8s} {'grad_pcc':>9s}", flush=True)
    for name in order:
        out_c = _cell(cpu_fwd, tt_fwd, name)
        grad_c = _cell(cpu_grad, tt_grad, name)
        if out_c.strip() == "." and grad_c.strip() == ".":
            continue
        print(f"[hooks] {name:50s} {out_c:>8s} {grad_c:>9s}", flush=True)


def _debug_residual_prefix_pcc(cfg, transformer, cpu_model, *, cpu_in, tt_in):
    """Per-depth output PCC on the COMPILED path (the real training path).

    Recompiles the transformer truncated to the first k blocks for k=1..N and
    compares the TT vs CPU final prediction at each depth. Because every depth
    goes through the same dynamo+tt compile (composites/fusion intact), this
    avoids the eager + low-signal confounds of leaf hooks. The drop from k-1 to
    k isolates how much block k degrades TT/CPU agreement.

    Gated by WAN_DEBUG_PREFIX=1. Recompiles N times, so it is slow; set
    WAN_PREFIX_STRIDE to sample fewer depths.
    """
    import torch_xla

    dev = _devmgr()
    x_t, timestep, text = cpu_in
    x_t_tt, timestep_tt, text_tt = tt_in

    tt_blocks = list(transformer.blocks)
    cpu_blocks = list(cpu_model.blocks)
    n = len(tt_blocks)
    stride = max(1, int(os.environ.get("WAN_PREFIX_STRIDE", "1")))
    depths = sorted(set(list(range(1, n + 1, stride)) + [n]))

    print(f"[prefix] per-depth compiled-path output PCC (n={n}, stride={stride})", flush=True)
    prev = None
    try:
        with torch.no_grad():
            for k in depths:
                transformer.blocks = nn.ModuleList(tt_blocks[:k])
                cpu_model.blocks = nn.ModuleList(cpu_blocks[:k])
                # Same module id but different block count -> must recompile.
                dev._compile_cache.pop(id(transformer), None)
                compiled = dev.compile(transformer)
                pred_tt = compiled(
                    hidden_states=x_t_tt, timestep=timestep_tt,
                    encoder_hidden_states=text_tt, return_dict=True,
                ).sample
                torch_xla.sync(wait=True)
                pred_cpu = cpu_model(
                    hidden_states=x_t, timestep=timestep,
                    encoder_hidden_states=text, return_dict=True,
                ).sample.float()
                pcc = _pcc(pred_tt, pred_cpu)
                delta = "" if prev is None else f"  d={prev - pcc:+.4f}"
                print(f"[prefix] depth={k:2d}  PCC(pred_tt,pred_cpu)={pcc:.4f}{delta}", flush=True)
                prev = pcc
    finally:
        transformer.blocks = nn.ModuleList(tt_blocks)
        cpu_model.blocks = nn.ModuleList(cpu_blocks)
        dev._compile_cache.pop(id(transformer), None)


class _Tap(nn.Module):
    """Transparent wrapper that records its inner module's output into a shared
    sink list, so the output becomes a real graph output (compile-safe). When
    the owner is in record mode (eager), it also appends this tap's name, giving
    a runtime-true ordering without having to predict execution order."""

    def __init__(self, inner: nn.Module, owner: "_TapTransformer", name: str,
                 record_input: bool = False):
        super().__init__()
        self.inner = inner
        self._owner = owner
        self._name = name
        self._record_input = record_input

    def forward(self, *args, **kwargs):
        if self._record_input:
            x = args[0] if args else next(iter(kwargs.values()))
            self._owner._sink.append(x)
            if self._owner._record:
                self._owner._names.append(self._name)
        out = self.inner(*args, **kwargs)
        if not self._record_input:
            self._owner._sink.append(out[0] if isinstance(out, tuple) else out)
            if self._owner._record:
                self._owner._names.append(self._name)
        return out

    def __getattr__(self, name):
        # Delegate unknown attrs (e.g. PEFT reads lora_A.weight.dtype) to inner.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)


def _wrap_lora_children(lin: nn.Module, prefix: str, owner: "_TapTransformer") -> None:
    """Wrap a PEFT lora.Linear's lora_A/lora_B adapters in place (if present)."""
    if hasattr(lin, "lora_A"):
        for adapter in list(lin.lora_A.keys()):
            lin.lora_A[adapter] = _Tap(lin.lora_A[adapter], owner, f"{prefix}.lora_A.{adapter}")
            lin.lora_B[adapter] = _Tap(lin.lora_B[adapter], owner, f"{prefix}.lora_B.{adapter}")


def _wrap_attn_leaf(attn: nn.Module, prefix: str, owner: "_TapTransformer") -> None:
    """Wrap the leaf sub-ops of a WanAttention: q/k/v proj (+their lora_A/lora_B),
    norm_q/norm_k, and to_out.0 (+its lora). Execution order inside the processor
    is to_q,to_k,to_v then norm_q,norm_k then to_out.0 — but ordering is captured
    at runtime via record mode, so we don't depend on guessing it here."""
    for proj in ("to_q", "to_k", "to_v"):
        lin = getattr(attn, proj)
        _wrap_lora_children(lin, f"{prefix}.{proj}", owner)
        setattr(attn, proj, _Tap(lin, owner, f"{prefix}.{proj}"))
    attn.norm_q = _Tap(attn.norm_q, owner, f"{prefix}.norm_q")
    attn.norm_k = _Tap(attn.norm_k, owner, f"{prefix}.norm_k")
    out0 = attn.to_out[0]
    _wrap_lora_children(out0, f"{prefix}.to_out.0", owner)
    attn.to_out[0] = _Tap(out0, owner, f"{prefix}.to_out.0")


class _TapAttnProcessor:
    """Decomposed Wan attention processor that taps the backward-relevant
    intermediates (q/k/v post-rope, scores, probs, attn_out) as graph outputs.

    Mirrors WanAttnProcessor math but replaces the fused SDPA with explicit
    matmul+softmax+matmul so the intermediates exist as tensors. autograd.grad
    on these then yields dQ/dK/dV, dScores (=dS), dP (=grad of probs) per
    attention, isolating exactly where TT/CPU grad PCC drops across the softmax
    jacobian. No I2V/add_k_proj branch (this model has none).
    """

    def __init__(self, owner: "_TapTransformer", prefix: str):
        self._owner = owner
        self._prefix = prefix

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, rotary_emb=None, **kwargs):
        owner = self._owner

        def tap(t, name):
            owner._sink.append(t)
            if owner._record:
                owner._names.append(f"{self._prefix}.{name}")
            return t

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        # Tap the projection output (pre-norm, pre-rope): this is exactly the
        # grad_out operand of the lora_B weight-grad contraction. Then post-norm
        # (pre-rope). Comparing q_proj -> q_norm -> q (post-rope) grads localizes
        # whether the drop is in rope-bwd, norm-bwd, or the contraction.
        q_proj = tap(attn.to_q(hidden_states), "q_proj")
        k_proj = tap(attn.to_k(encoder_hidden_states), "k_proj")
        query = tap(attn.norm_q(q_proj), "q_norm")
        key = tap(attn.norm_k(k_proj), "k_norm")
        value = tap(attn.to_v(encoder_hidden_states), "v_proj")

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            def apply_rotary_emb(hs, freqs_cos, freqs_sin):
                x1, x2 = hs.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hs)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hs)
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # [B, S, H, D] -> [B, H, S, D]
        q = tap(query.transpose(1, 2), "q")
        k = tap(key.transpose(1, 2), "k")
        v = tap(value.transpose(1, 2), "v")

        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = tap(torch.matmul(q, k.transpose(-1, -2)) * scale, "scores")
        probs = tap(torch.softmax(scores, dim=-1), "probs")
        attn_out = tap(torch.matmul(probs, v), "attn_out")

        out = attn_out.transpose(1, 2).flatten(2, 3).type_as(query)
        out = tap(attn.to_out[0](out), "out_proj")
        out = attn.to_out[1](out)
        return out


class _TapTransformer(nn.Module):
    """Wrap a WanTransformer3DModel so intermediate outputs are returned as real
    outputs (flat tuple: (pred, *taps)). Because they are graph outputs they
    survive torch.compile -> trustworthy per-module PCC on the compiled path,
    and torch.autograd.grad(loss, taps) gives the matching per-module gradient.

    leaf=False (default): coarse taps per block (attn1/attn2/ffn/block).
    leaf=True (WAN_TAP_LEAF=1): fine taps inside each attention — q/k/v proj and
    their lora_A/lora_B, norm_q/norm_k, to_out.0 — so you can see exactly which
    sub-op (e.g. before vs after norm_q, or base vs lora_B) degrades the PCC.

    Tap order/names are captured at runtime in record mode (set on the eager
    reference run), so no execution-order guessing is needed.
    """

    def __init__(self, transformer: WanTransformer3DModel, leaf: bool = False,
                 attn_decompose: bool = False):
        super().__init__()
        self.transformer = transformer
        self._sink: list = []
        self._names: list[str] = []
        self._record: bool = False
        orig = list(transformer.blocks)
        if attn_decompose:
            # Swap in a decomposed processor per attention that taps q/k/v,
            # scores, probs, attn_out. Blocks/ffn are left untouched so the
            # only graph outputs are the attention internals.
            for i, blk in enumerate(orig):
                blk.attn1.set_processor(_TapAttnProcessor(self, f"blocks.{i}.attn1"))
                blk.attn2.set_processor(_TapAttnProcessor(self, f"blocks.{i}.attn2"))
            return
        for i, blk in enumerate(orig):
            if leaf:
                _wrap_attn_leaf(blk.attn1, f"blocks.{i}.attn1", self)
                _wrap_attn_leaf(blk.attn2, f"blocks.{i}.attn2", self)
                blk.ffn = _Tap(blk.ffn, self, f"blocks.{i}.ffn")
                orig[i] = _Tap(blk, self, f"blocks.{i}")
            else:
                blk.attn1 = _Tap(blk.attn1, self, f"blocks.{i}.attn1")
                blk.attn2 = _Tap(blk.attn2, self, f"blocks.{i}.attn2")
                blk.ffn = _Tap(blk.ffn, self, f"blocks.{i}.ffn")
                orig[i] = _Tap(blk, self, f"blocks.{i}")
        transformer.blocks = nn.ModuleList(orig)
        # Head taps: the ~1.4% pred over-scaling enters between the last block
        # and the prediction. Split the head into norm_out (final LayerNorm,
        # pre-modulation) and proj_out (post (1+scale)+shift modulation + matmul)
        # so out_ratio localizes it to the LayerNorm vs the modulation/proj.
        ce = getattr(transformer, "condition_embedder", None)
        if ce is not None and getattr(ce, "time_embedder", None) is not None:
            # temb feeds the head modulation scale = scale_shift_table + temb.
            ce.time_embedder = _Tap(ce.time_embedder, self, "head.temb")
        if getattr(transformer, "norm_out", None) is not None:
            transformer.norm_out = _Tap(transformer.norm_out, self, "head.norm_out")
        if getattr(transformer, "proj_out", None) is not None:
            # Record proj_out INPUT (post-modulation hidden) AND output, so we
            # can tell whether (1+scale)+shift modulation injected the 1.4%
            # (proj_in already hot) or proj_out matmul did (proj_in clean).
            transformer.proj_out = _Tap(
                _Tap(transformer.proj_out, self, "head.proj_out"),
                self, "head.proj_in", record_input=True)

    def forward(self, **kwargs):
        self._sink = []
        if self._record:
            self._names = []
        out = self.transformer(**kwargs)
        pred = out.sample if hasattr(out, "sample") else out
        return (pred, *self._sink)


def _debug_tap_pcc(cfg, transformer, cpu_model, *, cpu_in, tt_in):
    """Per-module output + grad PCC on the COMPILED path via graph-output taps.

    Wraps both the TT transformer (compiled) and the CPU replica (eager
    reference) so each intermediate output is a real output, then compares
    output PCC and grad PCC (via autograd.grad) per module. Unlike eager forward
    hooks, this measures the real fused/composite compiled path.

    WAN_TAP_LEAF=1 taps inside each attention (q/k/v + lora_A/lora_B, norm_q/
    norm_k, to_out.0) for sub-op granularity; otherwise coarse per-block taps.
    Gated by WAN_DEBUG_TAP=1. Mutates the models in place (probe exits after).
    """
    import torch_xla

    dev = _devmgr()
    leaf = os.environ.get("WAN_TAP_LEAF", "0") == "1"
    attn_decompose = os.environ.get("WAN_TAP_ATTN", "0") == "1"
    x_t, timestep, text, target = cpu_in
    x_t_tt, timestep_tt, text_tt, target_tt = tt_in

    for p in cpu_model.parameters():
        p.grad = None
    for p in transformer.parameters():
        p.grad = None

    def _grad_over(loss, taps):
        # Some taps (e.g. head.temb) have no trainable params upstream and don't
        # require grad; autograd.grad rejects the whole list if any such tensor
        # is present, so differentiate only the grad-requiring ones and pad the
        # rest with None to keep alignment with names/taps.
        grads = [None] * len(taps)
        idx = [i for i, t in enumerate(taps)
               if isinstance(t, torch.Tensor) and t.requires_grad]
        if idx:
            g = torch.autograd.grad(
                loss, [taps[i] for i in idx], retain_graph=True, allow_unused=True)
            for i, gi in zip(idx, g):
                grads[i] = gi
        return grads

    # ----- CPU eager tap run (reference; records true tap order/names) -----
    cpu_tap = _TapTransformer(cpu_model, leaf=leaf, attn_decompose=attn_decompose)
    cpu_tap._record = True
    cpu_out = cpu_tap(
        hidden_states=x_t, timestep=timestep,
        encoder_hidden_states=text, return_dict=True,
    )
    cpu_pred, cpu_taps = cpu_out[0], list(cpu_out[1:])
    cpu_loss = F.mse_loss(cpu_pred.float(), target.float())
    cpu_grads = _grad_over(cpu_loss, cpu_taps)
    names = list(cpu_tap._names)

    # ----- TT compiled tap run (same arch -> same tap order) -----
    tt_tap = _TapTransformer(transformer, leaf=leaf, attn_decompose=attn_decompose)
    compiled = dev.compile(tt_tap)
    tt_out = compiled(
        hidden_states=x_t_tt, timestep=timestep_tt,
        encoder_hidden_states=text_tt, return_dict=True,
    )
    tt_pred, tt_taps = tt_out[0], list(tt_out[1:])
    tt_loss = F.mse_loss(tt_pred.float(), target_tt.float())
    tt_grads = _grad_over(tt_loss, tt_taps)
    torch_xla.sync(wait=True)

    if not (len(names) == len(tt_taps) == len(cpu_taps)):
        print(f"[tap] WARNING tap count mismatch: names={len(names)} "
              f"cpu={len(cpu_taps)} tt={len(tt_taps)} — order may be unreliable",
              flush=True)

    def cell(a, b) -> str:
        if a is None or b is None:
            return "   .  "
        a = a.to("cpu", torch.float32)
        b = b.to("cpu", torch.float32)
        if a.numel() != b.numel() or (a.norm() == 0 and b.norm() == 0):
            return "   .  "
        return f"{_pcc(a, b):+.4f}"

    def ratio(a, b) -> str:
        # std(tt)/std(cpu): PCC-blind scale gain. ~1.0 = no scale error; >1
        # means TT amplitude is inflated at this tap. Localizes the 1.36% gain.
        if a is None or b is None:
            return "  .  "
        a = a.to("cpu", torch.float32)
        b = b.to("cpu", torch.float32)
        if a.numel() != b.numel():
            return "  .  "
        sc = float(b.std())
        if sc == 0:
            return "  .  "
        return f"{float(a.std()) / sc:.4f}"

    w = max(22, *(len(n) for n in names)) if names else 22
    print(f"[tap] pred out PCC={_pcc(tt_pred, cpu_pred):.4f}  leaf={int(leaf)}  "
          f"attn={int(attn_decompose)}", flush=True)

    # ----- Loss-gap decomposition: PCC ~1 does NOT imply equal MSE -----
    # Loss is MSE, which (unlike PCC) is NOT scale/shift invariant. A tiny
    # systematic scale (a) or bias (b) in the TT prediction, or an uncorrelated
    # bf16 forward-noise floor, raises loss while leaving PCC ~0.9999. Decompose:
    #   loss_tt - loss_cpu = 2*<r_cpu, d>/N + mean(d^2)
    # with d = pred_tt - pred_cpu, r_cpu = pred_cpu - target.
    #   * mean(d^2)        -> uncorrelated TT-vs-CPU noise floor (always >=0)
    #   * 2*<r_cpu,d>/N    -> systematic part (scale/bias aligned with residual)
    # Also fit pred_tt ~= a*pred_cpu + b to expose scale (a!=1) / offset (b!=0).
    pc = cpu_pred.detach().to("cpu", torch.float32).flatten()
    pt = tt_pred.detach().to("cpu", torch.float32).flatten()
    tg = target.detach().to("cpu", torch.float32).flatten()
    tg_tt = target_tt.detach().to("cpu", torch.float32).flatten()
    N = pc.numel()
    # Exact pred-only decomposition on a MATCHED target (cpu target for both),
    # so dloss_matched == cross + floor holds exactly.
    loss_cpu_m = float(((pc - tg) ** 2).mean())
    loss_tt_m = float(((pt - tg) ** 2).mean())
    d = pt - pc
    r_cpu = pc - tg
    cross = 2.0 * float((r_cpu * d).sum()) / N
    floor = float((d * d).mean())
    # Target-side contribution: how much of the *reported* dloss is just the
    # bf16-vs-fp32 target (latent) difference, independent of the model.
    dtgt = tg_tt - tg
    tgt_term = float(((pt - tg_tt) ** 2).mean()) - loss_tt_m
    # least-squares fit pred_tt = a*pred_cpu + b
    pc_mean, pt_mean = float(pc.mean()), float(pt.mean())
    var_pc = float(((pc - pc_mean) ** 2).mean())
    cov = float(((pc - pc_mean) * (pt - pt_mean)).mean())
    a = cov / (var_pc + 1e-30)
    b = pt_mean - a * pc_mean
    resid_after_fit = float(((pt - (a * pc + b)) ** 2).mean())
    print(f"[gap] reported: loss_tt={float(tt_loss):.6f} loss_cpu={float(cpu_loss):.6f} "
          f"dloss={float(tt_loss) - float(cpu_loss):+.6f}", flush=True)
    print(f"[gap] matched-target: loss_tt={loss_tt_m:.6f} loss_cpu={loss_cpu_m:.6f} "
          f"dloss={loss_tt_m - loss_cpu_m:+.6f} (== cross+floor)", flush=True)
    print(f"[gap] decomp: cross(systematic)={cross:+.3e} "
          f"floor(noise=mean d^2)={floor:+.3e} target_term={tgt_term:+.3e}", flush=True)
    print(f"[gap] fit pred_tt ~= a*pred_cpu + b: a={a:.6f} (a-1={a-1:+.3e}) "
          f"b={b:+.3e}  resid_after_fit(MSE)={resid_after_fit:.3e}", flush=True)
    print(f"[gap] stats: mean(pt)={pt_mean:+.4e} mean(pc)={pc_mean:+.4e} "
          f"std(pt)={float(pt.std()):.4e} std(pc)={float(pc.std()):.4e} "
          f"||d||/||pc||={float(d.norm()/(pc.norm()+1e-30)):.3e} "
          f"||dtgt||/||tg||={float(dtgt.norm()/(tg.norm()+1e-30)):.3e}", flush=True)
    print(f"[tap] {'module':{w}s} {'out_pcc':>8s} {'out_ratio':>9s} {'grad_pcc':>9s}", flush=True)
    for i, nm in enumerate(names):
        o = cell(cpu_taps[i], tt_taps[i])
        r = ratio(cpu_taps[i], tt_taps[i])
        g = cell(cpu_grads[i], tt_grads[i])
        print(f"[tap] {nm:{w}s} {o:>8s} {r:>9s} {g:>9s}", flush=True)
    # Head (final norm_out + proj_out): the per-block taps stop at the last
    # residual, so compare the last block output's scale to the final pred to
    # see if the 1.36% gain is injected by the AdaLN head rather than the blocks.
    print(f"[tap] {'>> pred (head out)':{w}s} "
          f"{cell(cpu_pred, tt_pred):>8s} {ratio(cpu_pred, tt_pred):>9s} {'.':>9s}",
          flush=True)

    # ----- Head diagnostics: is the 1.4% from different weights, or a real
    # compute/sharding difference? std-ratio is muddied by the additive shift,
    # so also report the shift-robust slope a = cov(cpu,tt)/var(cpu). -----
    def slope(a_t, b_t):
        if a_t is None or b_t is None:
            return None
        a_t = a_t.to("cpu", torch.float32).flatten()
        b_t = b_t.to("cpu", torch.float32).flatten()
        if a_t.numel() != b_t.numel():
            return None
        am = a_t.mean()
        v = float(((a_t - am) ** 2).mean())
        if v == 0:
            return None
        return float(((a_t - am) * (b_t - b_t.mean())).mean()) / v

    head_idx = {nm: i for i, nm in enumerate(names) if nm.startswith("head.")}
    print("[head] ---- shift-robust scale (slope a = std_tt/std_cpu w/ offset removed) ----",
          flush=True)
    for nm in ("head.temb", "head.norm_out", "head.proj_in", "head.proj_out"):
        if nm in head_idx:
            s = slope(cpu_taps[head_idx[nm]], tt_taps[head_idx[nm]])
            print(f"[head] {nm:18s} slope_a={s:.6f} (a-1={s-1:+.3e})"
                  if s is not None else f"[head] {nm:18s}   .", flush=True)
    sp = slope(cpu_pred, tt_pred)
    print(f"[head] {'pred':18s} slope_a={sp:.6f} (a-1={sp-1:+.3e})", flush=True)

    # Weight equality: confirm CPU replica and TT use identical head weights, so
    # a non-zero scale can't be blamed on a weight mismatch (rules out the
    # simple structural cause; leaves real compute/sharding difference).
    def wdiff(name, getter):
        try:
            wc = getter(cpu_model).detach().to("cpu", torch.float32)
            wt = getter(transformer).detach().to("cpu", torch.float32)
        except Exception as e:  # noqa: BLE001
            print(f"[head] weight {name}: (unavailable: {e})", flush=True)
            return
        if wc.shape != wt.shape:
            print(f"[head] weight {name}: SHAPE MISMATCH {tuple(wc.shape)} vs {tuple(wt.shape)}",
                  flush=True)
            return
        mad = float((wc - wt).abs().max())
        rel = float((wc - wt).norm() / (wc.norm() + 1e-30))
        print(f"[head] weight {name:24s} max|Δ|={mad:.3e} relL2={rel:.3e}", flush=True)

    wdiff("scale_shift_table", lambda m: m.scale_shift_table)
    wdiff("proj_out.weight", lambda m: m.proj_out.weight)
    wdiff("proj_out.bias", lambda m: m.proj_out.bias)

    # ----- Apples-to-apples lora_B weight-grad PCC -----
    # Both cpu_model and transformer ran the SAME decomposed attention here, so
    # this isolates whether the ~0.84 q/k weight-grad PCC from the standard
    # probe is real or an artifact of comparing CPU fused-SDPA backward vs TT
    # decomposed backward. Also prints grad magnitude (|g_cpu|) and relative L2
    # error to test whether low PCC is just noise on tiny-magnitude grads.
    cpu_lb = {n: p for n, p in cpu_model.named_parameters()
              if p.requires_grad and "lora_B" in n}
    tt_lb = {n: p for n, p in transformer.named_parameters()
             if p.requires_grad and "lora_B" in n}
    lb_names = sorted(set(cpu_lb) & set(tt_lb))
    if lb_names:
        cpu_wg = torch.autograd.grad(
            cpu_loss, [cpu_lb[n] for n in lb_names],
            retain_graph=True, allow_unused=True)
        tt_wg = torch.autograd.grad(
            tt_loss, [tt_lb[n] for n in lb_names],
            retain_graph=True, allow_unused=True)
        torch_xla.sync(wait=True)
        lw = max(22, *(len(n) for n in lb_names))
        print("[wg] ---- lora_B weight-grad PCC (BOTH decomposed; apples-to-apples) ----",
              flush=True)
        print(f"[wg] {'param':{lw}s} {'pcc':>8s} {'|g_cpu|':>11s} {'relerr':>8s}", flush=True)
        for n, gc, gt in zip(lb_names, cpu_wg, tt_wg):
            if gc is None or gt is None:
                print(f"[wg] {n:{lw}s} {'   .  ':>8s}", flush=True)
                continue
            gc32 = gc.detach().to("cpu", torch.float32)
            gt32 = gt.detach().to("cpu", torch.float32)
            ncpu = float(gc32.norm())
            rel = float((gt32 - gc32).norm() / (ncpu + 1e-12))
            print(f"[wg] {n:{lw}s} {_pcc(gc32, gt32):>+8.4f} {ncpu:>11.3e} {rel:>8.3f}",
                  flush=True)


def _debug_multistep_pcc(cfg, transformer, compiled_transformer, cpu_model, *, cpu_in, tt_in, n_steps):
    """Run n_steps real optimizer updates on CPU replica vs compiled TT, on the
    SAME fixed batch each step, and report how loss/pred/param PCC drifts.

    A single step is ~0.9998 PCC; this checks whether per-step bf16 grad noise
    *compounds* into the run-level loss gap. Same fixed batch isolates the
    compounding from input variation. Gated by WAN_DEBUG_STEPS=<n>. Mutates the
    LoRA weights in place (probe exits after).
    """
    import torch_xla

    x_t, timestep, text, target = cpu_in
    x_t_tt, timestep_tt, text_tt, target_tt = tt_in

    cpu_params = [p for n, p in cpu_model.named_parameters() if p.requires_grad and "lora_" in n]
    tt_params = [p for n, p in transformer.named_parameters() if p.requires_grad and "lora_" in n]
    opt_kw = dict(lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.999), capturable=False)
    opt_cpu = torch.optim.AdamW(cpu_params, **opt_kw)
    opt_tt = torch.optim.AdamW(tt_params, **opt_kw)

    def _param_pcc() -> float:
        va, vb = _lora_param_vec(cpu_model), _lora_param_vec(transformer)
        keys = [k for k in va if k in vb]
        return _pcc(torch.cat([va[k] for k in keys]), torch.cat([vb[k] for k in keys]))

    verbose = os.environ.get("WAN_DEBUG_STEPS_VERBOSE", "0") == "1"

    cpu_model.train()
    transformer.train()
    print(f"[steps] multistep compounding test  n_steps={n_steps}  lr={cfg.LR}  "
          f"verbose={int(verbose)}", flush=True)
    print(f"[steps] {'step':>4s} {'loss_tt':>9s} {'loss_cpu':>9s} {'dloss':>8s} "
          f"{'pcc_pred':>9s} {'grad_min':>9s} {'grad_mean':>9s} {'pcc_param':>10s}", flush=True)
    for step in range(1, n_steps + 1):
        opt_cpu.zero_grad(set_to_none=True)
        pred_cpu = cpu_model(
            hidden_states=x_t, timestep=timestep,
            encoder_hidden_states=text, return_dict=True,
        ).sample.float()
        loss_cpu = F.mse_loss(pred_cpu, target)
        loss_cpu.backward()
        cpu_grads = _lora_grad_vec(cpu_model)
        opt_cpu.step()

        opt_tt.zero_grad(set_to_none=True)
        pred_tt = compiled_transformer(
            hidden_states=x_t_tt, timestep=timestep_tt,
            encoder_hidden_states=text_tt, return_dict=True,
        ).sample
        loss_tt = F.mse_loss(pred_tt.float(), target_tt.float())
        loss_tt.backward()
        tt_grads = _lora_grad_vec(transformer)
        opt_tt.step()
        torch_xla.sync(wait=True)

        # Per-LoRA grad PCC for this step (skip lora_A: grad==0 at init).
        gk = [k for k in cpu_grads if k in tt_grads]
        per_grad = {k: _pcc(cpu_grads[k], tt_grads[k]) for k in gk}
        nz = [v for v in per_grad.values() if not math.isnan(v)]
        gmin = min(nz) if nz else float("nan")
        gmean = (sum(nz) / len(nz)) if nz else float("nan")
        pred_pcc = _pcc(pred_tt.detach().to("cpu").float(), pred_cpu.detach())
        param_pcc = _param_pcc()
        dl = loss_tt.item() - loss_cpu.item()
        print(f"[steps] {step:>4d} {loss_tt.item():>9.4f} {loss_cpu.item():>9.4f} "
              f"{dl:>+8.4f} {pred_pcc:>9.4f} {gmin:>9.4f} {gmean:>9.4f} {param_pcc:>10.4f}",
              flush=True)
        if verbose:
            for k in sorted(per_grad):
                print(f"[steps]   s{step} grad {k}: PCC={per_grad[k]:.4f}", flush=True)


def _build_cpu_replica(cfg: Config, tt_transformer: WanTransformer3DModel) -> WanTransformer3DModel:
    """Same arch + same (already-LoRA'd) weights as the TT transformer, on CPU."""
    m = WanTransformer3DModel.from_pretrained(
        cfg.MODEL_ID, subfolder="transformer", torch_dtype=cfg.DTYPE, low_cpu_mem_usage=True,
    )
    _truncate_dit_blocks(m, cfg.DEBUG_DIT_BLOCKS)
    for p in m.parameters():
        p.requires_grad_(False)
    m.add_adapter(_make_lora_config(cfg))
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
    """One training step: forward + backward on CPU replica vs compiled TT (LoRA on)."""
    import torch_xla

    dev = _devmgr()
    B = batch["latent"].shape[0]
    x0 = batch["latent"].to(cfg.DTYPE)
    text = batch["text_embed"].to(cfg.DTYPE)
    t = _sample_timesteps(B, cfg).to(cfg.DTYPE)
    noise = torch.randn(x0.shape, dtype=x0.dtype)
    sigma = t.view(B, 1, 1, 1, 1)
    x_t = (1.0 - sigma) * x0 + sigma * noise
    timestep = (t * 1000.0).long()
    target = (noise - x0).float()

    print(
        f"[debug] LoRA probe  blocks={cfg.DEBUG_DIT_BLOCKS}  "
        f"same weights as train compiled_transformer  "
        f"x_t={tuple(x_t.shape)}  timestep={tuple(timestep.shape)} {timestep.dtype}  "
        f"target={tuple(target.shape)}",
        flush=True,
    )

    cpu_model = _build_cpu_replica(cfg, transformer)
    cpu_model.train()

    def _zero_grads(m: nn.Module) -> None:
        for p in m.parameters():
            if p.grad is not None:
                p.grad = None

    # ----- CPU forward + backward -----
    _zero_grads(cpu_model)
    pred_cpu = cpu_model(
        hidden_states=x_t,
        timestep=timestep,
        encoder_hidden_states=text,
        return_dict=True,
    ).sample.float()
    loss_cpu = F.mse_loss(pred_cpu, target)
    loss_cpu.backward()
    cpu_lora_grads = _lora_grad_vec(cpu_model)

    # ----- TT forward + backward (training compile path) -----
    transformer.train()
    _zero_grads(transformer)

    x0_tt = dev.to_device(x0)
    text_tt = dev.to_device(text)
    t_tt = dev.to_device(t)
    noise_tt = dev.to_device(noise)
    sigma_tt = t_tt.view(B, 1, 1, 1, 1)
    x_t_tt = (1.0 - sigma_tt) * x0_tt + sigma_tt * noise_tt
    timestep_tt = (t_tt * 1000.0).long()
    target_tt = noise_tt - x0_tt

    pred_tt = compiled_transformer(
        hidden_states=x_t_tt,
        timestep=timestep_tt,
        encoder_hidden_states=text_tt,
        return_dict=True,
    ).sample
    loss_tt = F.mse_loss(pred_tt.float(), target_tt.float())
    loss_tt.backward()
    torch_xla.sync(wait=True)
    pred_tt_cpu = pred_tt.detach().to("cpu").float()
    tt_lora_grads = _lora_grad_vec(transformer)

    print(
        f"[debug] FORWARD  loss_tt={loss_tt.item():.4f}  loss_cpu={loss_cpu.item():.4f}  "
        f"PCC(pred_tt,pred_cpu)={_pcc(pred_tt_cpu, pred_cpu):.4f}  "
        f"PCC(pred_tt,target)={_pcc(pred_tt_cpu, target):.4f}  "
        f"PCC(pred_cpu,target)={_pcc(pred_cpu, target):.4f}",
        flush=True,
    )
    _print_lora_grad_pcc(cpu_lora_grads, tt_lora_grads)

    # Multistep compounding test: N real optimizer updates on the same fixed
    # batch, tracking how loss/pred/param PCC drift step-over-step. Mutates LoRA
    # weights in place -> run before the other (gated) probes below.
    _n_steps = int(os.environ.get("WAN_DEBUG_STEPS", "0"))
    if _n_steps > 0:
        _debug_multistep_pcc(
            cfg, transformer, compiled_transformer, cpu_model,
            cpu_in=(x_t, timestep, text, target),
            tt_in=(x_t_tt, timestep_tt, text_tt, target_tt),
            n_steps=_n_steps,
        )

    # Localize where TT/CPU numerics diverge per leaf op (forward activation +
    # backward grad). NOTE: runs the EAGER device path, which does not match the
    # compiled training path — keep for quick looks, prefer WAN_DEBUG_PREFIX.
    if os.environ.get("WAN_DEBUG_HOOKS", "0") == "1":
        _compare_layer_pccs(
            cfg, transformer, cpu_model,
            cpu_in=(x_t, timestep, text, target),
            tt_in=(x_t_tt, timestep_tt, text_tt, target_tt),
        )

    # Per-depth output PCC on the COMPILED path (recompiles per block-prefix).
    if os.environ.get("WAN_DEBUG_PREFIX", "0") == "1":
        _debug_residual_prefix_pcc(
            cfg, transformer, cpu_model,
            cpu_in=(x_t, timestep, text),
            tt_in=(x_t_tt, timestep_tt, text_tt),
        )

    # Per-module output + grad PCC on the COMPILED path via graph-output taps
    # (single compile, real hook semantics). Mutates models in place -> run last.
    if os.environ.get("WAN_DEBUG_TAP", "0") == "1":
        _debug_tap_pcc(
            cfg, transformer, cpu_model,
            cpu_in=(x_t, timestep, text, target),
            tt_in=(x_t_tt, timestep_tt, text_tt, target_tt),
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
        loss = flow_matching_step(transformer, batch, cfg, fixed_t=t, fixed_noise=noise)
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


class _CmpDump:
    """Collect named tensors during a videogen run and save them to one .pt so
    a TT run and a CPU/GPU run can be diffed offline (PCC / atol / rtol).

    Enabled by `WAN_DUMP_DIR`; the file is tagged by `WAN_DUMP_TAG` (tt/cpu/cuda).
    All tensors are moved to CPU fp32 (vae_out kept fp16 to save space)."""

    def __init__(self, path: str):
        self.path = path
        self.store: dict[str, torch.Tensor] = {}

    def add(self, key: str, t: torch.Tensor, dtype=torch.float32) -> None:
        self.store[key] = t.detach().to("cpu", dtype).contiguous()

    def save(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.store, self.path)
        print(f"[dump] wrote {len(self.store)} tensors -> {self.path}", flush=True)


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
    if dev.use_tt:
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

    # WAN_DUMP_DIR enables the TT-vs-reference tensor dump (no Chisel). Every
    # captured tensor is saved to one .pt tagged by WAN_DUMP_TAG (e.g. tt/cpu/
    # cuda). Run once on TT and once on a non-TT device (WAN_INFER_DEVICE), then
    # diff offline with `_cmp_runs.py`.
    rec = None
    _dump_dir = os.environ.get("WAN_DUMP_DIR", "")
    if _dump_dir:
        rec = _CmpDump(os.path.join(_dump_dir, f"dump_{os.environ.get('WAN_DUMP_TAG', 'run')}.pt"))
        rec.add("prompt_embeds", prompt_embeds)
        if negative_prompt_embeds is not None:
            rec.add("negative_prompt_embeds", negative_prompt_embeds)
        rec.add("init_latents", latents)

    # WAN_TAP_DUMP=1: in addition to the noise_*/latents dump, capture per-block
    # (coarse: attn1/attn2/ffn/block + head) outputs of the DiT for BOTH the cond
    # and uncond forward, keyed by tap name. Run once on TT and once on CPU, then
    # diff per layer offline with `_cmp_runs.py` to see exactly which block's
    # output (and the cond-vs-uncond delta) first diverges. These taps survive
    # torch.compile because _TapTransformer returns them as real graph outputs.
    # Heavy: combine with WAN_INFER_MAX_STEPS=1 and a small --infer-frames.
    tap_compiled = None
    _tap = None
    if rec is not None and os.environ.get("WAN_TAP_DUMP", "") == "1":
        # WAN_TAP_ATTN=1 swaps the fused SDPA for an explicit matmul+softmax+
        # matmul processor that taps q/k/v (post-rope), scores, probs, attn_out
        # per attention. Compare TT-vs-CPU on these to see whether the fused SDPA
        # lowering is what attenuates the cross-attention output (if decomposed
        # attn_out matches CPU, the fused path is the bug).
        _attn_dec = os.environ.get("WAN_TAP_ATTN", "0") == "1"
        _tap = _TapTransformer(
            pipe.transformer, leaf=os.environ.get("WAN_TAP_LEAF", "0") == "1",
            attn_decompose=_attn_dec)
        _tap._record = True
        tap_compiled = dev.compile(_tap)
        print("[infer] WAN_TAP_DUMP=1: per-block cond/uncond tap dump enabled "
              f"(leaf={os.environ.get('WAN_TAP_LEAF', '0')}, "
              f"attn_decompose={_attn_dec})", flush=True)
    _tap_dtype = torch.float16 if os.environ.get("WAN_TAP_FP16", "1") == "1" else torch.float32

    # WAN_INFER_MAX_STEPS caps the denoise loop (0 = all). Set to 1 to capture
    # ONLY the first videogen DiT forward (Chisel per-op at the full ~6630-token
    # inference sequence) without the 40-step rollout or VAE decode.
    max_steps = int(os.environ.get("WAN_INFER_MAX_STEPS", "0") or 0)
    _n_steps = len(timesteps)
    for _i, t in enumerate(timesteps):
        _t_step = time.time()
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

        _cond_taps = _uncond_taps = None
        with _cache_ctx(pipe.transformer, "cond"):
            if tap_compiled is not None:
                _o = tap_compiled(
                    hidden_states=lat_dev, timestep=ts_dev,
                    encoder_hidden_states=prompt_embeds, return_dict=True)
                noise_cond, _cond_taps = _o[0], list(_o[1:])
            else:
                noise_cond = compiled_transformer(
                    hidden_states=lat_dev,
                    timestep=ts_dev,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
        noise_cond = cpu_cast(noise_cond)        # sync -> per-step graph boundary

        noise_uncond = None
        if do_cfg:
            with _cache_ctx(pipe.transformer, "uncond"):
                if tap_compiled is not None:
                    _o = tap_compiled(
                        hidden_states=lat_dev, timestep=ts_dev,
                        encoder_hidden_states=negative_prompt_embeds, return_dict=True)
                    noise_uncond, _uncond_taps = _o[0], list(_o[1:])
                else:
                    noise_uncond = compiled_transformer(
                        hidden_states=lat_dev,
                        timestep=ts_dev,
                        encoder_hidden_states=negative_prompt_embeds,
                        return_dict=False,
                    )[0]
            noise_uncond = cpu_cast(noise_uncond)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = noise_cond

        if rec is not None:
            rec.add(f"step{_i:03d}_timestep", timestep.float())
            rec.add(f"step{_i:03d}_latent_in", latent_model_input)
            rec.add(f"step{_i:03d}_noise_cond", noise_cond)
            if noise_uncond is not None:
                rec.add(f"step{_i:03d}_noise_uncond", noise_uncond)
            rec.add(f"step{_i:03d}_noise_pred", noise_pred)

        if tap_compiled is not None and rec is not None and _cond_taps is not None:
            names = list(_tap._names)
            if len(names) != len(_cond_taps):
                print(f"[infer] WARNING tap name/count mismatch: names={len(names)} "
                      f"taps={len(_cond_taps)} (keys may be unreliable)", flush=True)
            for nm, tv in zip(names, _cond_taps):
                rec.add(f"step{_i:03d}_cond::{nm}", tv, dtype=_tap_dtype)
            if _uncond_taps is not None:
                for nm, tv in zip(names, _uncond_taps):
                    rec.add(f"step{_i:03d}_unc::{nm}", tv, dtype=_tap_dtype)

        # scheduler step on CPU keeps UniPC's per-step scalars out of the DiT graph.
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if rec is not None:
            rec.add(f"step{_i:03d}_latents_out", latents)

        print(f"[infer] step {_i + 1}/{_n_steps}  t={float(t):.1f}  "
              f"noise_pred(std={noise_pred.float().std():.4f})  "
              f"{time.time() - _t_step:.1f}s", flush=True)

        if max_steps and (_i + 1) >= max_steps:
            print(f"[infer] WAN_INFER_MAX_STEPS={max_steps} reached; stopping after "
                  f"step {_i + 1} (first-step capture). Skipping VAE decode.")
            if rec is not None:
                rec.save()
            return None

    if os.environ.get("WAN_SKIP_VAE", "") == "1":
        print("[infer] WAN_SKIP_VAE=1; skipping VAE decode.")
        if rec is not None:
            rec.save()
        return None

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
    if rec is not None:
        rec.add("vae_in", latents_vae)

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
    if rec is not None:
        rec.add("vae_out", video, dtype=torch.float16)
        rec.save()
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


def _cpu_flow_matching_step(
    model: nn.Module, batch: dict, cfg: Config,
    t: torch.Tensor, noise: torch.Tensor,
    return_pred: bool = False,
) -> torch.Tensor:
    """CPU mirror of flow_matching_step: SAME math, no device transfer, and the
    timestep/noise are passed in so TT and CPU see identical inputs."""
    x0 = batch["latent"].to("cpu", cfg.DTYPE)
    text_embed = batch["text_embed"].to("cpu", cfg.DTYPE)
    B = x0.shape[0]
    t = t.to("cpu", cfg.DTYPE)
    noise = noise.to("cpu", x0.dtype)
    sigma = t.view(B, 1, 1, 1, 1)
    timestep = (t * 1000.0).long()
    x_t = (1.0 - sigma) * x0 + sigma * noise
    pred = model(
        hidden_states=x_t,
        timestep=timestep,
        encoder_hidden_states=text_embed,
        return_dict=True,
    ).sample
    target = noise - x0
    loss = F.mse_loss(pred.float(), target.float())
    if return_pred:
        return loss, pred
    return loss


def _affine_fit(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, float]:
    """Least-squares fit y ≈ a*x + b. Returns (a, b, norm_ratio=||y||/||x||).
    All on CPU float32, flattened. Used to separate a multiplicative scale (a)
    from an additive bias (b) in pred_tt vs pred_cpu."""
    x = x.detach().to("cpu", torch.float32).flatten()
    y = y.detach().to("cpu", torch.float32).flatten()
    mx, my = x.mean(), y.mean()
    vx = x - mx
    denom = (vx * vx).sum()
    a = float(((vx * (y - my)).sum() / denom)) if denom > 0 else float("nan")
    b = float(my - a * mx)
    nr = float(y.norm() / x.norm()) if x.norm() > 0 else float("nan")
    return a, b, nr


def compare_train(cfg: Config):
    """TT-vs-CPU training comparison. Mirrors train() exactly (same dataloader,
    flow_matching_step, grad-accum, AdamW, xm.optimizer_step) but runs a CPU
    replica in lockstep on the SAME batches + SAME noise/timesteps. Per micro-
    step it prints loss_tt vs loss_cpu and the LoRA grad PCC.

    8 micro-steps with GRAD_ACCUM=4 -> 2 optimizer steps.
    """
    import torch_xla

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
    train_indices = all_indices[:-cfg.VAL_HOLDOUT]
    print(f"[compare] {len(train_indices)} train cached samples")

    embeds = torch.load(embeds_path, weights_only=False)
    train_ds = PixelArtLatentDataset(cfg.CACHE_DIR, train_indices)
    train_collate = make_collate_fn(embeds, p_drop=cfg.TEXT_DROP_PROB, seed=cfg.SEED)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH, shuffle=True, collate_fn=train_collate,
        num_workers=0, drop_last=True,
    )

    # TT (compiled) model + a CPU replica with the SAME (LoRA'd) weights.
    transformer = build_lora_transformer(cfg)
    transformer.train()
    compiled_transformer = _devmgr().compile(transformer)
    cpu_model = _build_cpu_replica(cfg, transformer)
    cpu_model.train()

    trainable_tt = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_tt, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.999),
        capturable=False,
    )
    cpu_params = [p for n, p in cpu_model.named_parameters() if p.requires_grad and "lora_" in n]
    opt_cpu = torch.optim.AdamW(
        cpu_params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY, betas=(0.9, 0.999),
        capturable=False,
    )

    n_optim_steps = int(os.environ.get("WAN_OPT_STEPS", "2"))
    n_micro = n_optim_steps * cfg.GRAD_ACCUM
    print(
        f"[compare] blocks={cfg.DEBUG_DIT_BLOCKS}  micro_steps={n_micro}  "
        f"accum={cfg.GRAD_ACCUM}  optim_steps={n_optim_steps}  "
        f"batch={cfg.BATCH}  lr={cfg.LR}  dtype={cfg.DTYPE}",
        flush=True,
    )

    # Pin the input stream so it's INDEPENDENT of model depth/param count.
    # Building the LoRA adapters (gaussian init) and the shuffled dataloader both
    # consume the global RNG by an amount that scales with DEBUG_DIT_BLOCKS, which
    # would otherwise give every depth a different batch/timestep/noise and make a
    # depth sweep meaningless. Re-seed right before creating the iterator (fixes
    # the shuffle permutation) and draw t/noise from a dedicated generator.
    torch.manual_seed(cfg.SEED)
    input_gen = torch.Generator()
    input_gen.manual_seed(cfg.SEED)

    data_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    opt_cpu.zero_grad(set_to_none=True)
    global_step = 0
    for micro_step in range(1, n_micro + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Sample shared timestep + noise ONCE on CPU; feed to both paths so the
        # only difference is the device numerics (same as train(): noise/t on CPU).
        # Use the fixed `input_gen` so these are identical across depths.
        B = batch["latent"].shape[0]
        t = _sample_timesteps(B, cfg, generator=input_gen)
        noise = torch.randn(batch["latent"].shape, dtype=cfg.DTYPE, generator=input_gen)

        # ----- TT path (identical to train()) -----
        loss_tt, pred_tt = flow_matching_step(
            compiled_transformer, batch, cfg, fixed_t=t, fixed_noise=noise,
            return_pred=True)
        (loss_tt / cfg.GRAD_ACCUM).backward()
        _devmgr().sync()

        # ----- CPU path (same math, on CPU replica) -----
        loss_cpu, pred_cpu = _cpu_flow_matching_step(
            cpu_model, batch, cfg, t, noise, return_pred=True)
        (loss_cpu / cfg.GRAD_ACCUM).backward()

        # Pred-level (loss-independent) device discrepancy: PCC + affine fit
        # pred_tt ≈ a*pred_cpu + b. `a` is the multiplicative gain, `b` the bias,
        # nr=||pred_tt||/||pred_cpu||. This is the clean accumulation metric.
        pred_pcc = _pcc(pred_tt, pred_cpu)
        a, b, nr = _affine_fit(pred_cpu, pred_tt)

        lt, lc = loss_tt.item(), loss_cpu.item()
        print(
            f"[compare] micro {micro_step:>2d}/{n_micro}  "
            f"loss_tt={lt:.4f}  loss_cpu={lc:.4f}  dloss={lt - lc:+.4f}  "
            f"pred_pcc={pred_pcc:.5f}  a={a:.5f}  b={b:+.4e}  nr={nr:.5f}",
            flush=True,
        )
        # Accumulated LoRA grads so far (running sum within the accum window).
        _print_lora_grad_pcc(_lora_grad_vec(cpu_model), _lora_grad_vec(transformer))

        if micro_step % cfg.GRAD_ACCUM == 0:
            xm.optimizer_step(optimizer, barrier=True)
            optimizer.zero_grad()
            _devmgr().sync()
            opt_cpu.step()
            opt_cpu.zero_grad(set_to_none=True)
            global_step += 1
            param_va = _lora_param_vec(cpu_model)
            param_vb = _lora_param_vec(transformer)
            pkeys = [k for k in param_va if k in param_vb]
            ppcc = _pcc(
                torch.cat([param_va[k] for k in pkeys]),
                torch.cat([param_vb[k] for k in pkeys]),
            )
            print(
                f"[compare] === optimizer step {global_step}/{n_optim_steps} done  "
                f"LoRA param PCC(cpu,tt)={ppcc:.4f} ===",
                flush=True,
            )

    print(f"[compare] done: {global_step} optimizer steps over {n_micro} micro-steps", flush=True)


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
    resume_requested = bool(cfg.RESUME_PATH) and Path(cfg.RESUME_PATH).exists()
    if resume_requested:
        resume_lora(transformer, cfg.RESUME_PATH)
    elif cfg.RESUME_PATH:
        print(f"[train] RESUME_PATH set but not found ({cfg.RESUME_PATH}); starting fresh")
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

    # Restore optimizer state + step + RNG if a train-state file accompanies the
    # resume checkpoint; else fall back to RESUME_STEP (weights-only resume).
    resume_step = 0
    if resume_requested:
        loaded_step = load_train_state(optimizer, cfg.RESUME_PATH)
        resume_step = loaded_step if loaded_step is not None else cfg.RESUME_STEP
        print(f"[train] resumed weights from {cfg.RESUME_PATH} at step {resume_step}")

    run_name = f"wan22_5b_pxa_r{cfg.LORA_RANK}a{cfg.LORA_ALPHA}_{int(time.time() % 1_000_000)}"
    logger = Logger(cfg.WANDB_ENABLED, cfg.WANDB_PROJECT, run_name, cfg.asdict())

    # ----- Pre-training baseline (step=0) videogen: DISABLED. Videogen is the
    # step that crashed; training-only run. Re-enable by uncommenting below.
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
    print("[train] videogen disabled; training-only run")

    global_step = resume_step
    ema_loss: float | None = None
    ema_alpha = 0.1
    accum_loss = 0.0
    accum_count = 0
    micro_step = 0
    step_start = time.time()
    data_iter = iter(train_loader)

    print(f"[train] starting loop: max_steps={cfg.MAX_STEPS}, accum={cfg.GRAD_ACCUM}, "
          f"batch={cfg.BATCH}, lr={cfg.LR}, wd={cfg.WEIGHT_DECAY}")
    optimizer.zero_grad(set_to_none=True)

    # ----- DEBUG probe DISABLED: run the real training loop (1 step) under the
    # chisel session instead of the TT-vs-CPU comparison probe. Re-enable below.
    # _debug_first_step_tt_vs_cpu(cfg, transformer, compiled_transformer, next(data_iter))
    # exit(0)

    while global_step < cfg.MAX_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        loss = flow_matching_step(compiled_transformer, batch, cfg)
        (loss / cfg.GRAD_ACCUM).backward()
        # Force a sync after backward so loss.item() below doesn't trigger N
        # implicit small graph compiles per step.
        _devmgr().sync()
        accum_loss += loss.item()
        accum_count += 1
        micro_step += 1
        #print(f"[train loss after 1 step: {loss.item():.4f}]", flush=True)
        #exit(0)

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
            # DEBUG: stop after a single completed optimizer step.
            #exit(0)
            log_payload = {
                "train/loss": avg_loss,
                "train/loss_ema": ema_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/step_time_s": step_time,
            }
            logger.log(log_payload, step=global_step)
            accum_loss = 0.0
            accum_count = 0

            if cfg.VAL_LOSS_EVERY > 0 and global_step % cfg.VAL_LOSS_EVERY == 0:
                vloss = validation_loss(compiled_transformer, val_loader, cfg)
                logger.log({"val/loss": vloss}, step=global_step)
                print(f"[train] step {global_step}: val/loss={vloss:.4f}")

            if global_step % cfg.VAL_IMG_EVERY == 0:
                # Save an in-progress LoRA checkpoint. (Video gen DISABLED —
                # training-only run. Re-enable the block below to restore it.)
                ckpt_path = str(Path(cfg.LORA_PATH).with_name(
                    Path(cfg.LORA_PATH).stem + f"_step{global_step:05d}.safetensors"))
                save_lora(transformer, ckpt_path)
                save_train_state(optimizer, global_step, ckpt_path)
                # try:
                #     pil, vid = generate_validation_sample(transformer, cfg, global_step)
                #     logger.log_image(
                #         "val/sample", pil, step=global_step,
                #         caption=f"step={global_step} prompt={cfg.TRIGGER + cfg.VAL_PROMPT!r}",
                #     )
                #     if vid is not None:
                #         logger.log_video("val/sample_video", vid, fps=cfg.INFER_FPS, step=global_step)
                # except Exception as e:
                #     print(f"[val-img] step {global_step}: generation failed ({e!r}); "
                #           f"training continues, checkpoint at {ckpt_path}")

    save_lora(transformer, cfg.LORA_PATH)
    save_train_state(optimizer, global_step, cfg.LORA_PATH)
    print(f"[train] done at step {global_step}; saved LoRA to {cfg.LORA_PATH}")

    # Final video generation DISABLED — training-only run. Re-enable below.
    # infer(cfg, cfg.LORA_PATH, cfg.VAL_PROMPT, logger=logger, transformer=transformer)
    logger.finish()


# ---------------------------------------------------------------------------
# 11. Save LoRA
# ---------------------------------------------------------------------------


def resume_lora(transformer: WanTransformer3DModel, path: str) -> None:
    """Load LoRA weights from a checkpoint into the live (already-adapted)
    transformer. Mirror of `save_lora`: strips the saved "transformer." prefix
    that `save_lora` adds for diffusers' pipeline loader, then writes the
    weights into the existing PEFT adapter via `set_peft_model_state_dict`.
    """
    from safetensors.torch import load_file

    sd = load_file(path)
    # save_lora prefixes every key with "transformer."; PEFT expects bare keys.
    sd = {k[len("transformer."):] if k.startswith("transformer.") else k: v
          for k, v in sd.items()}
    # Move onto the transformer's device so mark_sharding annotations carry over.
    device = next(transformer.parameters()).device
    sd = {k: v.to(device=device, dtype=cfg_dtype(transformer)) for k, v in sd.items()}
    result = set_peft_model_state_dict(transformer, sd)
    missing = getattr(result, "unexpected_keys", None)
    print(f"[resume] loaded {len(sd)} LoRA tensors from {path}"
          + (f" (unexpected={list(missing)})" if missing else ""))


def cfg_dtype(transformer: WanTransformer3DModel):
    for p in transformer.parameters():
        if p.requires_grad:
            return p.dtype
    return next(transformer.parameters()).dtype


def _move_to_cpu(obj):
    """Recursively move tensors in a (nested) state dict to CPU for torch.save.
    Optimizer state lives on the XLA device; serialize it from host memory."""
    if torch.is_tensor(obj):
        return obj.detach().to("cpu")
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_cpu(v) for v in obj)
    return obj


def _train_state_path(lora_path: str) -> Path:
    """Sibling train-state file for a LoRA checkpoint: foo.safetensors -> foo.opt.pt."""
    return Path(lora_path).with_suffix(".opt.pt")


def save_train_state(optimizer: torch.optim.Optimizer, global_step: int, lora_path: str):
    """Persist optimizer state + step + RNG next to the LoRA checkpoint so a
    crash can resume exactly. The LoRA .safetensors stays the inference artifact;
    this .opt.pt is the training-resume artifact."""
    import torch_xla
    torch_xla.sync(wait=True)  # materialize lazy optimizer-state tensors first
    path = _train_state_path(lora_path)
    state = {
        "global_step": global_step,
        "optimizer": _move_to_cpu(optimizer.state_dict()),
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[save] wrote optimizer/train state (step={global_step}) -> {path}")


def load_train_state(optimizer: torch.optim.Optimizer, lora_path: str) -> int | None:
    """Restore optimizer state + step + RNG saved by `save_train_state`. Returns
    the saved global_step, or None if no train-state file exists (weights-only
    resume — optimizer starts cold)."""
    path = _train_state_path(lora_path)
    if not path.exists():
        print(f"[resume] no train-state file at {path}; optimizer starts fresh (cold Adam)")
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    # Optimizer.load_state_dict casts floating-point state to each param's
    # device/dtype, so the CPU tensors land back on the XLA device automatically.
    optimizer.load_state_dict(state["optimizer"])
    rng = state.get("rng", {})
    if "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "python" in rng:
        random.setstate(rng["python"])
    step = int(state.get("global_step", 0))
    print(f"[resume] restored optimizer/train state from {path} (step={step})")
    return step


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
    if video is None:
        print("[infer] no video produced (step-capped Chisel run); done.")
        return
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
    sp.add_parser("compare", parents=[common], help="TT-vs-CPU loss/LoRA-grad PCC over a few steps")
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
        # WAN_CHISEL_MODE=isolated|accumulated selects the Chisel check mode;
        # WAN_CHISEL_OUT overrides the results path (defaults to chisel_<mode>.jsonl).
        _mode = os.environ.get("WAN_CHISEL_MODE", "accumulated").lower()
        if _mode in ("off", "none", "disabled", "0"):
            # Real training run: skip Chisel's per-op golden validation entirely
            # (it re-runs a golden for every op every step — far too slow for a
            # full MAX_STEPS run).
            print("[chisel] DISABLED (WAN_CHISEL_MODE=off) — full-speed training", flush=True)
            train(cfg)
        else:
            _iso = _mode in ("isolated", "isolation", "iso")
            checks_config = chisel.ChiselChecksConfig(
                isolation=_iso,
                accumulation=not _iso,
            )
            _out = os.environ.get("WAN_CHISEL_OUT", f"chisel_{'isolated' if _iso else 'accumulated'}.jsonl")
            print(f"[chisel] mode={'isolated' if _iso else 'accumulated'} -> {_out}", flush=True)
            with chisel.session(results_path=_out, checks_config=checks_config) as report:
                train(cfg)
    elif args.cmd == "compare":
        compare_train(cfg)
    elif args.cmd == "infer":
        lora_path = args.lora or cfg.LORA_PATH
        prompt = args.prompt or cfg.VAL_PROMPT
        # WAN_CHISEL_MODE=isolated|accumulated wraps inference in a Chisel
        # session (default off). Combine with WAN_INFER_MAX_STEPS=1 to capture
        # per-op PCC for ONLY the first videogen DiT forward.
        _mode = os.environ.get("WAN_CHISEL_MODE", "off").lower()
        if _mode in ("off", "none", "disabled", "0", ""):
            infer(cfg, lora_path, prompt,
                  output_path=args.output,
                  negative_prompt=args.negative_prompt)
        else:
            _iso = _mode in ("isolated", "isolation", "iso")
            checks_config = chisel.ChiselChecksConfig(
                isolation=_iso,
                accumulation=not _iso,
            )
            _out = os.environ.get(
                "WAN_CHISEL_OUT",
                f"chisel_infer_{'isolated' if _iso else 'accumulated'}.jsonl",
            )
            print(f"[chisel] infer mode={'isolated' if _iso else 'accumulated'} -> {_out}", flush=True)
            with chisel.session(results_path=_out, checks_config=checks_config) as report:
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
