# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import ml_dtypes
import numpy as np
import torch
from PIL import Image

from blacksmith.datasets.tt_train.omniconsistency_lego.omniconsistency_lego_dataset import (
    load_samples,
    strip_style_words,
)
from blacksmith.experiments.tt_train.wan2_2.configs import TrainingConfig
from blacksmith.experiments.tt_train.wan2_2.timing import (
    phase,
    record,
    set_sink,
    summary,
)
from blacksmith.models.tt_train.wan2_2.encoders import (
    WanTextEncoderTT,
    WanVAEEncoderTT,
    close_mesh,
    make_ccl_manager,
    open_mesh,
    ttnn_dtype,
)
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.tt_train.logging_manager import TrainingLogger

DEFAULT_CONFIG = Path(__file__).parent / "lora" / "galaxy" / "wan2_2_t2v_a14b_lego.yaml"


def _center_crop_resize(img: Image.Image, h: int, w: int) -> Image.Image:
    iw, ih = img.size
    target_ratio, src_ratio = w / h, iw / ih
    if src_ratio > target_ratio:
        new_w = int(round(ih * target_ratio))
        x0 = (iw - new_w) // 2
        img = img.crop((x0, 0, x0 + new_w, ih))
    else:
        new_h = int(round(iw / target_ratio))
        y0 = (ih - new_h) // 2
        img = img.crop((0, y0, iw, y0 + new_h))
    return img.resize((w, h), Image.LANCZOS)


def _pil_to_video_tensor(img: Image.Image, h: int, w: int, num_frames: int = 1) -> torch.Tensor:
    img = _center_crop_resize(img, h, w)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    t = t.unsqueeze(0).unsqueeze(2)
    if num_frames > 1:
        t = t.repeat(1, 1, num_frames, 1, 1)
    return t


def _validate_res(config: TrainingConfig, vae_config) -> None:
    spatial = 2 ** len(vae_config.temperal_downsample)
    multiple = spatial * 2
    for label, v in [("train_h", config.train_h), ("train_w", config.train_w)]:
        if v % multiple != 0:
            raise ValueError(
                f"{label}={v} must be a multiple of {multiple} (VAE spatial stride {spatial} * patch_size 2)."
            )


def precompute(config: TrainingConfig, logger: TrainingLogger) -> None:
    cache = Path(config.cache_dir)
    samples_dir = cache / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(config.data_dir)
    if config.subset_size:
        samples = samples[: config.subset_size]
    logger.info(f"{len(samples)} (image, caption) pairs from {config.data_dir}")

    logger.info(f"opening mesh device {tuple(config.vae_parallel_shape)} ...")
    with phase("open mesh"):
        mesh_device = open_mesh(config.vae_parallel_shape)
    try:
        ccl_manager = make_ccl_manager(mesh_device)

        logger.info(f"building Wan VAE encoder on device (dtype={config.vae_dtype}) ...")
        with phase("build VAE encoder"):
            vae = WanVAEEncoderTT(
                checkpoint_name=config.model_id,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                height=config.train_h,
                width=config.train_w,
                num_frames=config.train_frames,
                dtype=ttnn_dtype(config.vae_dtype),
            )
        _validate_res(config, vae.config)

        metadata: list[dict] = []
        logger.info(f"VAE-encoding {len(samples)} images at {config.train_h}x{config.train_w} ...")
        vae_start = time.perf_counter()
        for i, (img, caption) in enumerate(samples):
            video = _pil_to_video_tensor(img, config.train_h, config.train_w, config.train_frames)
            latent = vae.encode(video)
            if config.strip_style_words:
                caption = strip_style_words(caption)
            triggered = config.trigger + caption
            np.save(samples_dir / f"sample_{i:04d}.npy", latent.detach().float().cpu().numpy())
            metadata.append({"idx": i, "caption": triggered})
            if (i + 1) % 8 == 0 or i == len(samples) - 1:
                done = time.perf_counter() - vae_start
                logger.info(f"  {i + 1}/{len(samples)} latent.shape={tuple(latent.shape)} ({done / (i + 1):.2f}s/img)")
        record("VAE encode", time.perf_counter() - vae_start)

        (cache / "metadata.json").write_text(json.dumps(metadata, indent=2))
        del vae
        gc.collect()

        unique_captions = sorted({m["caption"] for m in metadata})
        if "" not in unique_captions:
            unique_captions.append("")

        logger.info("building UMT5 text encoder on device ...")
        with phase("build text encoder"):
            text_encoder = WanTextEncoderTT(
                checkpoint_name=config.model_id,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                max_sequence_length=config.max_seq,
            )
        logger.info(f"T5-encoding {len(unique_captions)} unique captions ...")
        with phase("T5 encode"):
            embeds = text_encoder.encode(unique_captions)
        del text_encoder
        gc.collect()
    finally:
        with phase("close mesh"):
            close_mesh(mesh_device)

    with phase("write cache"):
        captions = list(embeds.keys())
        table = np.stack([embeds[c].float().cpu().numpy().astype(ml_dtypes.bfloat16) for c in captions], 0)
        np.save(cache / "embeds.npy", table)
        (cache / "embeds_index.json").write_text(json.dumps({c: i for i, c in enumerate(captions)}, indent=2))
    logger.info(f"done. cache at {cache.resolve()} — {len(metadata)} samples, {len(captions)} embeds.")


if __name__ == "__main__":
    args = parse_cli_options(default_config=DEFAULT_CONFIG)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, overrides=args.overrides)

    logger = TrainingLogger(config.model_copy(update={"use_wandb": False}), args.test_log_filename_prefix)
    set_sink(logger.info)

    started = time.perf_counter()
    try:
        precompute(config, logger)
    finally:
        summary("precompute", time.perf_counter() - started)
        logger.finish()
