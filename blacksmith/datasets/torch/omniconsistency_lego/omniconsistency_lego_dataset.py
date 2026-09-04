# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_ART = r"(?:a\s+|an\s+|the\s+)?"
_STYLE_PHRASES = [
    r"inspired by[^,.]*",
    rf"in\s+{_ART}blocky\s+lego\s+form",
    r"blocky\s+lego\s+form",
    rf"in\s+{_ART}lego\s+style",
    r"lego\s+minifigure\s+art\s+style",
    r"lego\s+figure\s+style",
    r"lego\s+minifigure\s+style",
    r"lego\s+art\s+style",
    r"lego\s+minifigure",
    r"lego\s+style",
    r"blocky\s+shapes",
]
_PHRASE_RE = re.compile("|".join(_STYLE_PHRASES), re.I)
_STYLE_WORD_RE = re.compile(r"\b(lego|blocky|minifigure)\b", re.I)


def strip_style_words(caption: str) -> str:
    text = _PHRASE_RE.sub("", caption)
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        if _STYLE_WORD_RE.search(s):
            continue
        body, end = re.match(r"(.*?)([.!?]*)\s*$", s.strip()).groups()
        clauses = [c.strip() for c in body.split(",") if c.strip()]
        if clauses:
            out.append(", ".join(clauses) + end)
    return re.sub(r"\s{2,}", " ", " ".join(out)).strip(" ,")


def download_style_subset(dataset_id: str, style: str, data_dir: str) -> tuple[Path, int, int]:
    from huggingface_hub import HfApi, hf_hub_download

    out = Path(data_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    repo_files = HfApi().list_repo_files(dataset_id, repo_type="dataset")
    images = sorted(f for f in repo_files if f.startswith(f"{style}/tar/") and f.endswith(".png"))
    captions = {
        Path(f).stem for f in repo_files if f.startswith(f"{style}/caption/") and f.endswith(".txt")
    }
    if not images:
        available = sorted({f.split("/")[0] for f in repo_files if "/" in f})
        raise FileNotFoundError(
            f"no {style}/tar/*.png in {dataset_id}; available styles: {', '.join(available)}"
        )

    metadata: list[dict] = []
    skipped = 0
    for image_file in images:
        stem = Path(image_file).stem
        if stem not in captions:
            skipped += 1
            continue
        # Caption first: an empty one skips the sample, so there is no point fetching its image.
        caption_path = hf_hub_download(
            repo_id=dataset_id, filename=f"{style}/caption/{stem}.txt", repo_type="dataset"
        )
        caption = Path(caption_path).read_text(encoding="utf-8").strip()
        if not caption:
            skipped += 1
            continue
        image_path = hf_hub_download(repo_id=dataset_id, filename=image_file, repo_type="dataset")
        shutil.copyfile(image_path, images_dir / f"{stem}.png")
        metadata.append({"idx": int(stem), "image": f"images/{stem}.png", "caption": caption})

    metadata.sort(key=lambda m: m["idx"])
    (out / "metadata.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in metadata) + "\n", encoding="utf-8"
    )
    if not metadata:
        raise RuntimeError("no usable pairs produced")
    return out, len(metadata), skipped


def load_samples(data_dir: str) -> list[tuple[Image.Image, str]]:
    out = Path(data_dir)
    rows = [json.loads(line) for line in (out / "metadata.jsonl").read_text().splitlines() if line.strip()]
    return [(Image.open(out / r["image"]).convert("RGB"), r["caption"]) for r in rows]


def transform(img: Image.Image, h: int, w: int) -> Image.Image:
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


def pil_to_video_tensor(img: Image.Image, h: int, w: int, num_frames: int = 1) -> torch.Tensor:
    """PIL RGB -> (1, 3, num_frames, H, W) in [-1, 1] on CPU.

    `num_frames > 1` repeats the still into a static clip, so the adapter is trained on
    every temporal position rather than only the first frame.
    """
    img = transform(img, h, w)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # [0, 1] -> [-1, 1], the input range the Wan VAE encoder expects.
    arr = arr * 2.0 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    t = t.unsqueeze(0).unsqueeze(2)
    if num_frames > 1:
        t = t.repeat(1, 1, num_frames, 1, 1)
    return t


class LatentEmbedDataset(Dataset):

    def __init__(self, cache_dir: str, indices: list[int]) -> None:
        cache = Path(cache_dir)
        self.samples_dir = cache / "samples"
        self.indices = list(indices)
        meta = json.loads((cache / "metadata.json").read_text())
        self.captions = {m["idx"]: m["caption"] for m in meta}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, Any]:
        idx = self.indices[i]
        data = torch.load(self.samples_dir / f"sample_{idx:04d}.pt", weights_only=False)
        # Tolerate both a bare tensor and the {"latent", "caption"} dict the pixelart
        # precompute writes, so either cache layout loads.
        latent = data["latent"] if isinstance(data, dict) else data
        return {"latent": latent, "caption": self.captions[idx], "idx": idx}


def make_collate_fn(
    embeds: dict[str, torch.Tensor], p_drop: float, seed: int = 0
) -> Callable[[list[dict]], dict]:
    rng = random.Random(seed)

    def collate(batch: list[dict]) -> dict:
        latents, text_embeds, captions_used, idxs = [], [], [], []
        for item in batch:
            cap = "" if rng.random() < p_drop else item["caption"]
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
