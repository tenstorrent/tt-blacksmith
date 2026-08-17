# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import time
from contextlib import nullcontext
from pathlib import Path

from typing import TYPE_CHECKING

import numpy as np
import torch

from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig
from blacksmith.models.torch.wan2_2.model_overrides import (
    UMT5Wrapper,
    VAEDecoderWrapper,
    safe_xla_slicing,
)

if TYPE_CHECKING:
    # Type-only: the concrete manager is backend-specific and importing the tt-xla one
    # eagerly would drag `torch_xla` into the pure-torch (tt-kurbla) path.
    from blacksmith.models.torch.wan2_2.device import WanDeviceManager


def _cache_ctx(model, name: str):
    cc = getattr(model, "cache_context", None)
    if callable(cc):
        try:
            return cc(name)
        except Exception:
            return nullcontext()
    return nullcontext()


def _val_embed_cache_path(config: TrainingConfig) -> Path:
    return Path(config.cache_dir) / "val_prompt_embeds.pt"


def _val_embed_key(config: TrainingConfig, text: str, max_sequence_length: int) -> tuple:
    # The layer count is part of the key so a truncated (meaningless) embedding can never
    # be served to a full-depth run, or the reverse.
    return (text, max_sequence_length, config.text_encoder_layers, config.dtype)


def load_cached_val_embeds(config: TrainingConfig, texts, max_sequence_length: int):
    """The cached embeddings for `texts`, or None if any is missing.

    All-or-nothing: one miss means the text encoder has to be loaded anyway, and once it
    is loaded encoding the rest is comparatively free.
    """
    path = _val_embed_cache_path(config)
    if not config.cache_val_prompt_embeds or not path.exists():
        return None
    cached = torch.load(path, weights_only=False)
    keys = [_val_embed_key(config, t, max_sequence_length) for t in texts]
    if any(k not in cached for k in keys):
        return None
    print(f"[infer] using cached prompt embeddings from {path}", flush=True)
    return {t: cached[k] for t, k in zip(texts, keys)}


def _store_val_embeds(config: TrainingConfig, device_manager: "WanDeviceManager", embeds: dict, max_seq: int) -> None:
    if not config.cache_val_prompt_embeds:
        return
    path = _val_embed_cache_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = torch.load(path, weights_only=False) if path.exists() else {}
    for text, tensor in embeds.items():
        # gather() first: a DTensor written straight to disk keeps a device mesh that no
        # later process can reconstruct.
        stored[_val_embed_key(config, text, max_seq)] = device_manager.gather(tensor).to("cpu")
    torch.save(stored, path)
    print(f"[infer] cached {len(embeds)} prompt embedding(s) to {path}", flush=True)


def build_pipeline_for_validation(
    transformer, config: TrainingConfig, device_manager: "WanDeviceManager", *, need_text_encoder: bool = True
):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

    # This runs once per validation and reloads/re-shards the VAE and the 4.6B text
    # encoder every time, so it is a prime suspect whenever a validation feels slow.
    t_build = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(
        config.model_id, subfolder="vae", torch_dtype=config.torch_dtype(), low_cpu_mem_usage=True
    )
    # Backend-specific graph rewrites (a no-op on tt-xla). The VAE is constructed here,
    # so this is the only place a backend can get at it.
    vae = device_manager.prepare_model(vae)
    t_vae_load = time.perf_counter() - t_build
    t0 = time.perf_counter()
    # Passing text_encoder=None tells diffusers not to load that component at all, which
    # is the whole point of the embedding cache: the 4.6B encoder is never read from
    # disk, never moved to device and never sharded.
    text_encoder_kwargs = {} if need_text_encoder else {"text_encoder": None}
    pipe = WanPipeline.from_pretrained(
        config.model_id, transformer=transformer, vae=vae, torch_dtype=config.torch_dtype(), **text_encoder_kwargs
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=config.inference.infer_flow_shift
    )
    t_pipe_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    if getattr(pipe, "text_encoder", None) is not None:
        if config.text_encoder_layers is not None:
            blocks = pipe.text_encoder.encoder.block
            if config.text_encoder_layers < len(blocks):
                # Same bring-up trade as dit_layers, and done before the move so the
                # dropped blocks are never copied to device or sharded.
                print(
                    f"[infer] truncating the text encoder to the first "
                    f"{config.text_encoder_layers} of {len(blocks)} blocks",
                    flush=True,
                )
                pipe.text_encoder.encoder.block = blocks[: config.text_encoder_layers]
        # Move first, then re-tie: .to(device) copies each param, so tying before the
        # move would leave embed_tokens/shared as two independent device tensors.
        pipe.text_encoder = device_manager.to_device(pipe.text_encoder)
        pipe.text_encoder.encoder.embed_tokens.weight = pipe.text_encoder.shared.weight
        device_manager.shard_model(pipe.text_encoder)
    t_text = time.perf_counter() - t0

    t0 = time.perf_counter()
    pipe.vae = device_manager.to_device(pipe.vae)
    device_manager.shard_model(pipe.vae)
    print(
        f"[infer] pipeline built in {time.perf_counter() - t_build:.1f}s "
        f"(vae load {t_vae_load:.1f}s, pipeline load {t_pipe_load:.1f}s, "
        f"text encoder move+shard {t_text:.1f}s, vae move+shard {time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    return pipe


@torch.no_grad()
def generate_wan_video(
    pipe,
    compiled_transformer,
    config: TrainingConfig,
    device_manager: "WanDeviceManager",
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
    # Manual denoise loop: the compiled DiT is the only thing on TT inside the loop
    # (stable HLO -> compiled once), while scheduler/CFG/decode-prep stay on CPU.
    transformer_dtype = config.torch_dtype()
    do_cfg = guidance_scale > 1.0

    vae_t = pipe.vae.config.scale_factor_temporal
    vae_s = pipe.vae.config.scale_factor_spatial
    patch_size = pipe.transformer.config.patch_size

    if num_frames % vae_t != 1:
        num_frames = num_frames // vae_t * vae_t + 1
    num_frames = max(num_frames, 1)
    height = height // (vae_s * patch_size[1]) * (vae_s * patch_size[1])
    width = width // (vae_s * patch_size[2]) * (vae_s * patch_size[2])

    def tt_cast(x):
        # to_device, not a bare .to(device): with a mesh active every parameter of the
        # DiT/VAE is a DTensor, and an op mixing a plain tensor with a DTensor raises
        # ("got mixed torch.Tensor and DTensor"). to_device also replicates, which is
        # what .to("tt") already does physically. A no-op wrap when there is no mesh.
        return device_manager.to_device(x.to(transformer_dtype))

    def cpu_cast(x):
        # gather first: the scheduler step, CFG blend and `latents` below are plain CPU
        # tensors, so a DTensor coming back from the DiT would poison that arithmetic
        # with the same mixed-operand error. A no-op on plain tensors.
        return device_manager.gather(x).to(dtype=torch.float32, device="cpu")

    # Cache hit: the pipeline was built without a text encoder, so there is nothing to
    # compile and nothing to run -- just move the stored embeddings back to device.
    cached_embeds = load_cached_val_embeds(config, [prompt] + ([negative_prompt or ""] if do_cfg else []),
                                           max_sequence_length)
    if cached_embeds is not None:
        prompt_embeds = tt_cast(cached_embeds[prompt])
        negative_prompt_embeds = tt_cast(cached_embeds[negative_prompt or ""]) if do_cfg else None
        umt5 = None
    else:
        # UMT5 compiled as its own cached graph; wrapper stashed on `pipe` for a stable id().
        umt5 = getattr(pipe, "_compiled_umt5", None)
        if umt5 is None:
            pipe._umt5_wrapper = UMT5Wrapper(pipe.text_encoder)
            umt5 = device_manager.compile(pipe._umt5_wrapper)
            pipe._compiled_umt5 = umt5

    def _encode(text: str) -> torch.Tensor:
        tok = pipe.tokenizer(
            text, padding="max_length", truncation=True, max_length=max_sequence_length, return_tensors="pt"
        )
        input_ids = device_manager.to_device(tok.input_ids)
        attn_mask = device_manager.to_device(tok.attention_mask)
        out = umt5(input_ids, attn_mask)
        out = out * attn_mask.unsqueeze(-1).to(out.dtype)
        return out.to(transformer_dtype)

    if umt5 is not None:
        t_encode = time.perf_counter()
        prompt_embeds = _encode(prompt)
        negative_prompt_embeds = _encode(negative_prompt or "") if do_cfg else None
        print(
            f"[infer] text encode ({2 if do_cfg else 1} pass) in {time.perf_counter() - t_encode:.6f} seconds",
            flush=True,
        )
        to_cache = {prompt: prompt_embeds}
        if do_cfg:
            to_cache[negative_prompt or ""] = negative_prompt_embeds
        _store_val_embeds(config, device_manager, to_cache, max_sequence_length)
    # Flush the text-encoder graph before the denoise loop. On tt-xla this is the
    # `xm.mark_step()` that keeps UMT5 out of the DiT's graph; on tt-kurbla execution is
    # eager and `sync()` is a no-op.
    device_manager.sync()

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

    n_steps = len(timesteps)
    t_loop = time.perf_counter()
    step_times: list[float] = []
    for i, t in enumerate(timesteps):
        t_step = time.perf_counter()
        latent_model_input = latents.to(transformer_dtype)
        if expand_ts:
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(1, -1)
        else:
            timestep = t.expand(1)

        lat_dev = tt_cast(latent_model_input)
        ts_dev = device_manager.to_device(timestep)

        t_cond = time.perf_counter()
        with _cache_ctx(pipe.transformer, "cond"):
            noise_cond = compiled_transformer(
                hidden_states=lat_dev, timestep=ts_dev, encoder_hidden_states=prompt_embeds, return_dict=False
            )[0]
        noise_cond = cpu_cast(noise_cond)
        dur_cond = time.perf_counter() - t_cond

        t_uncond = time.perf_counter()
        if do_cfg:
            with _cache_ctx(pipe.transformer, "uncond"):
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

        dur_uncond = time.perf_counter() - t_uncond

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        dur = time.perf_counter() - t_step
        step_times.append(dur)
        # The first step carries the DiT's compile; the rest are steady-state.
        print(
            f"[infer] step {i + 1}/{n_steps} t={float(t):.1f} in {dur:.6f} seconds "
            f"(cond {dur_cond:.6f}s"
            + (f", uncond {dur_uncond:.6f}s" if do_cfg else "")
            + f", elapsed {time.perf_counter() - t_loop:.6f}s)",
            flush=True,
        )

    steady = step_times[1:] or step_times
    print(
        f"[infer] denoise loop: {n_steps} steps in {time.perf_counter() - t_loop:.6f} seconds "
        f"(first {step_times[0]:.6f}s incl. compile, steady-state mean {sum(steady) / len(steady):.6f}s)",
        flush=True,
    )

    if output_type == "latent":
        return latents

    latents_vae = latents.to(torch.float32)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents_vae.device, latents_vae.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(
        latents_vae.device, latents_vae.dtype
    )
    # Same reason as tt_cast: the VAE's parameters are DTensors under a mesh.
    latents_vae = device_manager.to_device((latents_vae / latents_std + latents_mean).to(pipe.vae.dtype))

    vae_decode = getattr(pipe, "_compiled_vae_decode", None)
    if vae_decode is None:
        pipe._vae_dec_wrapper = VAEDecoderWrapper(pipe.vae)
        vae_decode = device_manager.compile(pipe._vae_dec_wrapper)
        pipe._compiled_vae_decode = vae_decode

    t_decode = time.perf_counter()
    with safe_xla_slicing():
        video = vae_decode(latents_vae)
    # postprocess_video goes through numpy, which a DTensor cannot serve: gather here.
    video = device_manager.gather(video).to("cpu").to(torch.float32)
    print(f"[infer] vae decode in {time.perf_counter() - t_decode:.6f} seconds", flush=True)
    return pipe.video_processor.postprocess_video(video, output_type=output_type)


@torch.no_grad()
def generate_validation_sample(transformer, config: TrainingConfig, device_manager: "WanDeviceManager", step: int):
    # Returns (first_frame_PIL, all_frames_uint8_np_or_None).
    transformer.eval()
    inf = config.inference
    # Decide before building the pipeline: with every prompt already cached, the text
    # encoder is never loaded. val_prompt/neg_prompt are constants, so after the first
    # validation this is a hit on every run.
    prompts = [config.trigger + inf.val_prompt]
    if inf.infer_guidance > 1.0:
        prompts.append(inf.neg_prompt or "")
    need_text_encoder = load_cached_val_embeds(config, prompts, 512) is None

    pipe = build_pipeline_for_validation(
        transformer, config, device_manager, need_text_encoder=need_text_encoder
    )
    compiled_transformer = device_manager.compile(transformer)
    gen = torch.Generator(device="cpu").manual_seed(config.seed)
    video = generate_wan_video(
        pipe,
        compiled_transformer,
        config,
        device_manager,
        prompt=config.trigger + inf.val_prompt,
        negative_prompt=inf.neg_prompt or None,
        height=inf.infer_h,
        width=inf.infer_w,
        num_frames=inf.val_img_frames,
        num_inference_steps=inf.val_img_steps,
        guidance_scale=inf.infer_guidance,
        generator=gen,
        output_type="pil",
    )
    frames = video[0]
    img = frames[0]
    video_np = None
    if len(frames) > 1:
        video_np = np.stack([np.asarray(f) for f in frames], axis=0).astype(np.uint8)
    del pipe.vae
    del pipe
    gc.collect()
    transformer.train()
    return img, video_np
