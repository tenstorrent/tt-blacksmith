# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch.utils.data import Dataset

from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig


def download_and_subset_dataset(config: TrainingConfig) -> list[tuple[Image.Image, str]]:
    # `datasets >= 4` dropped loader-script support, so read the parquet + zips directly.
    meta_path = hf_hub_download(repo_id=config.dataset_id, filename="metadata.parquet", repo_type="dataset")
    df = pd.read_parquet(meta_path)

    text_col = "prompt" if "prompt" in df.columns else "text"
    if text_col not in df.columns:
        raise RuntimeError(f"no prompt/text column in metadata; have {list(df.columns)}")
    if "image_name" not in df.columns or "part_id" not in df.columns:
        raise RuntimeError(f"unexpected metadata schema; have {list(df.columns)}")

    repo_files = HfApi().list_repo_files(config.dataset_id, repo_type="dataset")
    available_parts = set()
    for f in repo_files:
        if f.startswith("images/part-") and f.endswith(".zip"):
            try:
                available_parts.add(int(Path(f).stem.split("-")[-1]))
            except ValueError:
                pass
    if not available_parts:
        raise RuntimeError(f"no images/part-*.zip files in {config.dataset_id}")

    df = df[df["part_id"].isin(available_parts)].reset_index(drop=True)
    df = df[df[text_col].astype(str).str.strip() != ""].reset_index(drop=True)
    df = df.sample(frac=1.0, random_state=config.seed).reset_index(drop=True)

    samples: list[tuple[Image.Image, str]] = []
    open_zips: dict[int, zipfile.ZipFile] = {}
    try:
        for _, row in df.iterrows():
            if len(samples) >= config.subset_size:
                break
            part_id = int(row["part_id"])
            if part_id not in open_zips:
                zip_name = f"images/part-{part_id:06d}.zip"
                try:
                    zip_path = hf_hub_download(repo_id=config.dataset_id, filename=zip_name, repo_type="dataset")
                except Exception:
                    continue
                open_zips[part_id] = zipfile.ZipFile(zip_path, "r")
            zf = open_zips[part_id]
            img_name = str(row["image_name"])
            try:
                with zf.open(img_name) as fp:
                    img = Image.open(io.BytesIO(fp.read())).convert("RGB")
            except KeyError:
                continue
            samples.append((img, str(row[text_col]).strip()))
    finally:
        for zf in open_zips.values():
            zf.close()

    if not samples:
        raise RuntimeError("no usable samples retrieved from dataset")
    return samples


def center_crop_resize(img: Image.Image, h: int, w: int) -> Image.Image:
    iw, ih = img.size
    target_ratio = w / h
    src_ratio = iw / ih
    if src_ratio > target_ratio:
        new_w = int(round(ih * target_ratio))
        x0 = (iw - new_w) // 2
        img = img.crop((x0, 0, x0 + new_w, ih))
    else:
        new_h = int(round(iw / target_ratio))
        y0 = (ih - new_h) // 2
        img = img.crop((0, y0, iw, y0 + new_h))
    return img.resize((w, h), Image.LANCZOS)


def pil_to_video_tensor(img: Image.Image, h: int, w: int) -> torch.Tensor:
    # PIL RGB -> (1, 3, 1, H, W) in [-1, 1] on CPU.
    img = center_crop_resize(img, h, w)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t.unsqueeze(0).unsqueeze(2)


def wan_latents_normalize(latents: torch.Tensor, vae) -> torch.Tensor:
    # Match the per-channel mean/std normalization WanPipeline applies.
    mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype, device=latents.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=latents.dtype, device=latents.device).view(1, -1, 1, 1, 1)
    return (latents - mean) * (1.0 / std)


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
        latents, text_embeds, captions_used, idxs = [], [], [], []
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
