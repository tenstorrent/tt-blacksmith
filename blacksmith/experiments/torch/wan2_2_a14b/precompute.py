# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from blacksmith.datasets.torch.omniconsistency_lego.omniconsistency_lego_dataset import (
    load_samples,
    pil_to_video_tensor,
    strip_style_words,
)
from blacksmith.experiments.torch.wan2_2_a14b.configs import TrainingConfig
from blacksmith.experiments.torch.wan2_2_a14b.kurbla.model_overrides import apply_generality_overrides
from blacksmith.tools.kurbla.device_manager import DeviceManager


class VAEEncoderWrapper(nn.Module):
    """Strip the diffusers output object so dynamo sees a plain tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return self.vae.encode(x).latent_dist.mode()


class UMT5Wrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def _validate_res(config: TrainingConfig, vae_config) -> None:
    """train_h/w must be a multiple of the VAE spatial stride times the DiT patch size."""
    spatial = 2 ** len(vae_config.temperal_downsample)
    multiple = spatial * 2
    for label, value in [("train_h", config.train_h), ("train_w", config.train_w)]:
        if value % multiple != 0:
            raise ValueError(
                f"{label}={value} must be a multiple of {multiple} "
                f"(VAE spatial stride {spatial} * DiT patch size 2)."
            )


def _normalize_latents(latent: torch.Tensor, vae, device_manager: DeviceManager) -> torch.Tensor:
    """Per-channel mean/std normalisation, as WanPipeline applies it."""
    mean = torch.tensor(vae.config.latents_mean, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    return (latent - device_manager.to_device(mean)) / device_manager.to_device(std)


def _dp_batch_size(device_manager: DeviceManager) -> int:
    """How many samples one encode call covers: the width of the data-parallel axis."""
    if not device_manager.is_data_parallel():
        return 1
    mesh = device_manager.mesh
    return mesh.size(mesh.mesh_dim_names.index(device_manager.input_sharding_dim))


@torch.no_grad()
def precompute_latents_and_embeds(config: TrainingConfig, device_manager: DeviceManager) -> None:
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel

    cache = Path(config.cache_dir)
    samples_dir = cache / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(config.data_dir)
    if config.subset_size:
        samples = samples[: config.subset_size]
    print(f"[precompute] {len(samples)} (image, caption) pairs from {config.data_dir}", flush=True)

    def triggered_caption(caption: str) -> str:
        if config.strip_style_words:
            caption = strip_style_words(caption)
        return config.trigger + caption

    metadata = [{"idx": i, "caption": triggered_caption(cap)} for i, (_, cap) in enumerate(samples)]
    todo = [i for i in range(len(samples)) if not (samples_dir / f"sample_{i:04d}.pt").exists()]

    dp_batch = _dp_batch_size(device_manager)

    if todo:
        vae_dtype = config.vae_torch_dtype()
        vae = AutoencoderKLWan.from_pretrained(
            config.model_id, subfolder="vae", torch_dtype=vae_dtype, low_cpu_mem_usage=True
        ).eval()
        _validate_res(config, vae.config)

        vae = device_manager.prepare_model(vae)
        vae = device_manager.to_device(vae)
        device_manager.shard_model(vae)
        vae_encode = device_manager.compile(VAEEncoderWrapper(vae))

        print(
            f"[precompute] VAE-encoding {len(todo)} of {len(samples)} images at "
            f"{config.train_h}x{config.train_w}x{config.train_frames}f, {dp_batch} at a time "
            f"({len(samples) - len(todo)} already cached) ...",
            flush=True,
        )
        started = time.perf_counter()
        done = 0
        for start in range(0, len(todo), dp_batch):
            chunk = todo[start : start + dp_batch]
            videos = [
                pil_to_video_tensor(samples[i][0], config.train_h, config.train_w, config.train_frames).to(vae_dtype)
                for i in chunk
            ]
            # Pad the last chunk: the DP axis must divide dim 0.
            video = torch.cat(videos + [videos[-1]] * (dp_batch - len(videos)), dim=0)

            latent = vae_encode(device_manager.prepare_batch({"video": video})["video"])
            latent = _normalize_latents(latent, vae, device_manager)
            # Gather before disk: a DTensor carries a mesh no later process can rebuild.
            latents = device_manager.gather(latent).to("cpu")

            for k, i in enumerate(chunk):
                sample = latents[k].contiguous()
                torch.save({"latent": sample, "caption": metadata[i]["caption"]}, samples_dir / f"sample_{i:04d}.pt")
                done += 1
                print(
                    f"[precompute]   {done}/{len(todo)} sample_{i:04d} {tuple(sample.shape)} "
                    f"({(time.perf_counter() - started) / done:.2f}s/img)",
                    flush=True,
                )
        del vae_encode, vae
        gc.collect()
    else:
        print(f"[precompute] all {len(samples)} latents already cached, skipping the VAE", flush=True)

    (cache / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # --- captions -----------------------------------------------------------------------
    unique_captions = sorted({m["caption"] for m in metadata})
    if "" not in unique_captions:
        unique_captions.append("")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        config.model_id, subfolder="text_encoder", torch_dtype=config.torch_dtype(), low_cpu_mem_usage=True
    ).eval()
    text_encoder = device_manager.prepare_model(text_encoder)
    text_encoder = device_manager.to_device(text_encoder)
    text_encoder.encoder.embed_tokens.weight = text_encoder.shared.weight
    device_manager.shard_model(text_encoder)
    umt5 = device_manager.compile(UMT5Wrapper(text_encoder))

    print(
        f"[precompute] T5-encoding {len(unique_captions)} unique captions, {dp_batch} at a time ...",
        flush=True,
    )
    embeds: dict[str, torch.Tensor] = {}
    started = time.perf_counter()
    done = 0
    for start in range(0, len(unique_captions), dp_batch):
        chunk = unique_captions[start : start + dp_batch]
        padded = chunk + [chunk[-1]] * (dp_batch - len(chunk))
        tok = tokenizer(
            padded, padding="max_length", truncation=True, max_length=config.max_seq, return_tensors="pt"
        )
        batch = device_manager.prepare_batch({"input_ids": tok.input_ids, "attention_mask": tok.attention_mask})
        input_ids, attn_mask = batch["input_ids"], batch["attention_mask"]

        out = umt5(input_ids, attn_mask)
        # Match WanPipeline.encode_prompt: zero the padding, keep the full length.
        out = out * attn_mask.unsqueeze(-1).to(out.dtype)
        gathered = device_manager.gather(out).to("cpu")

        for k, caption in enumerate(chunk):
            embeds[caption] = gathered[k].contiguous()
            done += 1
        print(
            f"[precompute]   {done}/{len(unique_captions)} "
            f"({(time.perf_counter() - started) / done:.2f}s/caption)",
            flush=True,
        )

    torch.save(embeds, cache / "embeds.pt")
    del umt5, text_encoder, tokenizer
    gc.collect()
    print(
        f"[precompute] done. cache at {cache.resolve()} — {len(metadata)} samples, {len(embeds)} embeds.",
        flush=True,
    )


if __name__ == "__main__":
    from blacksmith.tools.cli import generate_config, parse_cli_options
    from blacksmith.tools.reproducibility_manager import ReproducibilityManager

    DEFAULT_CONFIG = Path(__file__).parent / "kurbla" / "lora" / "galaxy" / "wan2_2_t2v_a14b_lego.yaml"
    args = parse_cli_options(default_config=DEFAULT_CONFIG)
    config: TrainingConfig = generate_config(TrainingConfig, args.config, args.test_config, overrides=args.overrides)

    ReproducibilityManager(config).setup()

    apply_generality_overrides()

    stage_config = config.model_copy(
        update={
            "mesh_shape": config.vae_parallel_shape,
            "input_sharding_dim": (
                config.mesh_axis_names[0] if config.vae_parallel_shape and config.mesh_axis_names else None
            ),
        }
    )
    device_manager = DeviceManager(stage_config)
    print(f"[precompute] device={device_manager.device} mesh={device_manager.mesh}", flush=True)

    precompute_latents_and_embeds(config, device_manager)
