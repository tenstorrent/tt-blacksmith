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

import ml_dtypes
import numpy as np
from PIL import Image

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
    from huggingface_hub import snapshot_download

    out = Path(data_dir)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    local = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        allow_patterns=[f"{style}/tar/*.png", f"{style}/caption/*.txt"],
    )
    style_root = Path(local) / style
    tar_dir, cap_dir = style_root / "tar", style_root / "caption"
    if not tar_dir.is_dir():
        raise FileNotFoundError(f"no {style}/tar/ in {dataset_id} — check the style name")

    metadata: list[dict] = []
    skipped = 0
    for img_path in sorted(tar_dir.glob("*.png")):
        stem = img_path.stem
        cap_path = cap_dir / f"{stem}.txt"
        if not cap_path.exists():
            skipped += 1
            continue
        caption = cap_path.read_text(encoding="utf-8").strip()
        if not caption:
            skipped += 1
            continue
        shutil.copyfile(img_path, images_dir / f"{stem}.png")
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


class LatentEmbedDataset:
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
        latent = np.load(self.samples_dir / f"sample_{idx:04d}.npy")
        return {"latent": latent, "caption": self.captions[idx], "idx": idx}


class TextEmbeds:
    def __init__(self, cache_dir: str) -> None:
        cache = Path(cache_dir)
        self.table = np.load(cache / "embeds.npy").view(ml_dtypes.bfloat16)
        self.index = json.loads((cache / "embeds_index.json").read_text())

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, caption: str) -> bool:
        return caption in self.index

    def __getitem__(self, caption: str) -> np.ndarray:
        return self.table[self.index[caption]]


def make_collate_fn(embeds: TextEmbeds, p_drop: float, seed: int = 0) -> Callable[[list[dict]], dict]:
    rng = random.Random(seed)

    def collate(examples: list[dict]) -> dict:
        latents, text_embeds, caps, idxs = [], [], [], []
        for item in examples:
            cap = "" if rng.random() < p_drop else item["caption"]
            if cap not in embeds:
                raise KeyError(f"missing precomputed embed for caption {cap!r}")
            latents.append(item["latent"])
            text_embeds.append(embeds[cap])
            caps.append(cap)
            idxs.append(item["idx"])
        return {
            "latent": np.stack(latents, 0).astype(np.float32),
            "text_embed": np.stack(text_embeds, 0).astype(np.float32),
            "captions": caps,
            "idx": idxs,
        }

    return collate
