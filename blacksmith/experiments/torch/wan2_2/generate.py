# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import time
from contextlib import nullcontext

import numpy as np
import torch

from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig
from blacksmith.models.torch.wan2_2.device import WanDeviceManager
from blacksmith.models.torch.wan2_2.model_overrides import (
    UMT5Wrapper,
    VAEDecoderWrapper,
    safe_xla_slicing,
)


def _cache_ctx(model, name: str):
    cc = getattr(model, "cache_context", None)
    if callable(cc):
        try:
            return cc(name)
        except Exception:
            return nullcontext()
    return nullcontext()


def build_pipeline_for_validation(transformer, config: TrainingConfig, device_manager: WanDeviceManager):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline

    vae = AutoencoderKLWan.from_pretrained(
        config.model_id, subfolder="vae", torch_dtype=config.torch_dtype(), low_cpu_mem_usage=True
    )
    pipe = WanPipeline.from_pretrained(
        config.model_id, transformer=transformer, vae=vae, torch_dtype=config.torch_dtype()
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=config.inference.infer_flow_shift
    )

    if getattr(pipe, "text_encoder", None) is not None:
        # Move first, then re-tie: .to(device) copies each param, so tying before the
        # move would leave embed_tokens/shared as two independent device tensors.
        pipe.text_encoder = device_manager.to_device(pipe.text_encoder)
        pipe.text_encoder.encoder.embed_tokens.weight = pipe.text_encoder.shared.weight
        device_manager.shard_model(pipe.text_encoder)
    pipe.vae = device_manager.to_device(pipe.vae)
    device_manager.shard_model(pipe.vae)
    return pipe


@torch.no_grad()
def generate_wan_video(
    pipe,
    compiled_transformer,
    config: TrainingConfig,
    device_manager: WanDeviceManager,
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
    device = device_manager.device
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
        return x.to(dtype=transformer_dtype, device=device)

    def cpu_cast(x):
        return x.to(dtype=torch.float32, device="cpu")

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

    prompt_embeds = _encode(prompt)
    negative_prompt_embeds = _encode(negative_prompt or "") if do_cfg else None
    if device_manager.config.use_tt:
        import torch_xla.core.xla_model as xm

        xm.mark_step()

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
    for i, t in enumerate(timesteps):
        t_step = time.time()
        latent_model_input = latents.to(transformer_dtype)
        if expand_ts:
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            timestep = temp_ts.unsqueeze(0).expand(1, -1)
        else:
            timestep = t.expand(1)

        lat_dev = tt_cast(latent_model_input)
        ts_dev = timestep.to(device)

        with _cache_ctx(pipe.transformer, "cond"):
            noise_cond = compiled_transformer(
                hidden_states=lat_dev, timestep=ts_dev, encoder_hidden_states=prompt_embeds, return_dict=False
            )[0]
        noise_cond = cpu_cast(noise_cond)

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

        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        print(f"[infer] step {i + 1}/{n_steps} t={float(t):.1f} {time.time() - t_step:.1f}s", flush=True)

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
    latents_vae = (latents_vae / latents_std + latents_mean).to(dtype=pipe.vae.dtype, device=device)

    vae_decode = getattr(pipe, "_compiled_vae_decode", None)
    if vae_decode is None:
        pipe._vae_dec_wrapper = VAEDecoderWrapper(pipe.vae)
        vae_decode = device_manager.compile(pipe._vae_dec_wrapper)
        pipe._compiled_vae_decode = vae_decode

    with safe_xla_slicing():
        video = vae_decode(latents_vae)
    video = video.to("cpu").to(torch.float32)
    return pipe.video_processor.postprocess_video(video, output_type=output_type)


@torch.no_grad()
def generate_validation_sample(transformer, config: TrainingConfig, device_manager: WanDeviceManager, step: int):
    # Returns (first_frame_PIL, all_frames_uint8_np_or_None).
    transformer.eval()
    pipe = build_pipeline_for_validation(transformer, config, device_manager)
    compiled_transformer = device_manager.compile(transformer)
    gen = torch.Generator(device="cpu").manual_seed(config.seed)
    inf = config.inference
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
