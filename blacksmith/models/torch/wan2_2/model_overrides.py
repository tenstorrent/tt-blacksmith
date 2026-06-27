# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig
from blacksmith.models.torch.wan2_2.device import WanDeviceManager


# --- Generality patches (correctness; the model will not run on TT without these) ---


def _patch_wan_resample_rep_sentinel() -> None:
    # Replace the "Rep" string sentinel with object() + is/is-not so a tensor slot
    # never triggers Tensor.__eq__(str) (which dynamo cannot trace). Subsumed by
    # _patch_wan_resample_avoid_4d_fold; applying both is harmless.
    try:
        from diffusers.models.autoencoders import autoencoder_kl_wan as akw
    except ImportError:
        return

    cache_t = akw.CACHE_T
    rep = object()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = rep
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -cache_t:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] is not rep:
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] is rep:
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] is rep:
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    akw.WanResample.forward = forward


_ORIG_GETITEM = torch.Tensor.__getitem__


def _clamp_slice(s: slice, size: int) -> slice:
    # Canonicalize a slice into bounds the way slice.indices(size) would, but with
    # plain int math (slice.indices is a slot wrapper dynamo graph-breaks on).
    start, stop, step = s.start, s.stop, s.step
    step = 1 if step is None else step

    if step > 0:
        if start is None:
            start = 0
        elif start < 0:
            start = max(0, start + size)
        else:
            start = min(start, size)
        if stop is None:
            stop = size
        elif stop < 0:
            stop = max(0, stop + size)
        else:
            stop = min(stop, size)
    else:
        if start is None:
            start = size - 1
        elif start < 0:
            start = max(-1, start + size)
        else:
            start = min(start, size - 1)
        if stop is None:
            stop = -1
        elif stop < 0:
            stop = max(-1, stop + size)
        else:
            stop = min(stop, size - 1)

    return slice(start, stop, step)


def _normalize_index(idx, shape):
    if not isinstance(idx, tuple):
        idx = (idx,)

    out = []
    dim = 0
    for item in idx:
        if item is Ellipsis:
            remaining_explicit = sum(x is not Ellipsis and x is not None for x in idx[idx.index(item) + 1 :])
            fill = len(shape) - dim - remaining_explicit
            out.extend([slice(None)] * fill)
            dim += fill
            continue
        if item is None:
            out.append(item)
            continue
        if isinstance(item, slice):
            out.append(_clamp_slice(item, shape[dim]))
            dim += 1
            continue
        out.append(item)
        dim += 1

    return tuple(out)


class _SafeSlicingMode(torch.overrides.TorchFunctionMode):
    # Clamp out-of-range slice indices in Tensor.__getitem__ before re-dispatch.
    # A TorchFunctionMode (not slot reassignment) keeps torch.tensor(list) fast path.
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.__getitem__:
            self_, idx = args
            return _ORIG_GETITEM(self_, _normalize_index(idx, self_.shape))
        return func(*args, **kwargs)


@contextmanager
def safe_xla_slicing():
    # CPU silently clamps slice start/stop outside [-size, size]; torch-xla raises.
    # AutoencoderKLWan relies on the CPU behavior, so wrap the whole VAE-decode run.
    with _SafeSlicingMode():
        yield


def _patch_wan_resample_avoid_4d_fold() -> None:
    # Replace the 5D->4D permute+reshape + nn.Upsample path with per-slice unbind /
    # repeat_interleave / Conv2d / stack so dim-1 channel sharding survives SPMD
    # (the nn.Upsample lowering produces wrong values on channel-sharded inputs).
    # Bit-equivalent to nearest-exact at 2x, and break-free for dynamo. Bakes in the
    # rep-sentinel fix.
    try:
        from diffusers.models.autoencoders import autoencoder_kl_wan as akw
    except ImportError:
        return

    cache_t = akw.CACHE_T
    rep = object()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = rep
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -cache_t:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] is not rep:
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] is rep:
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] is rep:
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)

        if self.mode in ("upsample2d", "upsample3d"):
            conv2d = self.resample[1]
            out_slices = []
            for s in torch.unbind(x, dim=2):
                s = s.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
                out_slices.append(conv2d(s))
            x = torch.stack(out_slices, dim=2)
        elif self.mode in ("downsample2d", "downsample3d"):
            t_now = x.shape[2]
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t_now, c, h, w)
            x = self.resample(x)
            x = x.view(b, t_now, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    akw.WanResample.forward = forward


# --- Perf patches (graph-break removals; no correctness impact) ---


def _patch_apply_lora_scale() -> None:
    # Make diffusers' apply_lora_scale a no-op (each scale/unscale is a graph break)
    # and rebind the already-decorated WanTransformer3DModel.forward to its unwrapped
    # body. The LoRA adapters themselves stay real; this only kills helper overhead.
    from diffusers.utils import peft_utils

    def noop_decorator(kwargs_name: str = "joint_attention_kwargs"):
        def decorator(forward_fn):
            return forward_fn

        return decorator

    peft_utils.apply_lora_scale = noop_decorator

    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    wrapped = WanTransformer3DModel.forward
    underlying = getattr(wrapped, "__wrapped__", None)
    if underlying is not None:
        WanTransformer3DModel.forward = underlying


def _patch_wan_time_embedder_dtype() -> None:
    # Replace the next(self.time_embedder.parameters()).dtype probe (graph break)
    # with a direct linear_1.weight.dtype read (always the first param, same dtype).
    from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding

    def forward(self, timestep, encoder_hidden_states, encoder_hidden_states_image=None, timestep_seq_len=None):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image

    WanTimeTextImageEmbedding.forward = forward


def _disable_tt_torch_function_override() -> None:
    # Pop tt_torch's import-time TorchFunctionMode off the stack; it is a no-op on
    # the compile path but forces a __torch_function__ trace on every matmul/linear.
    try:
        import tt_torch.torch_overrides as overrides
    except ImportError:
        return

    mode = getattr(overrides, "torch_function_override", None)
    if mode is None:
        return
    try:
        mode.__exit__(None, None, None)
    except Exception:
        pass


def apply_generality_overrides() -> None:
    _patch_wan_resample_rep_sentinel()
    _patch_wan_resample_avoid_4d_fold()


def apply_perf_overrides() -> None:
    _patch_apply_lora_scale()
    _patch_wan_time_embedder_dtype()
    _disable_tt_torch_function_override()


# --- Wrappers (strip diffusers return objects to a plain tensor for dynamo) ---


class UMT5Wrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class VAEEncoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return self.vae.encode(x).latent_dist.mode()


class VAEDecoderWrapper(nn.Module):
    # Compiling this unrolls AutoencoderKLWan's per-frame _decode loop into a single
    # graph (eager LTC recompiles each frame/chunk).
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


# --- LoRA transformer construction ---


def _make_lora_config(config: TrainingConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_targets),
        lora_dropout=0.0,
        init_lora_weights="gaussian",
    )


def build_lora_transformer(config: TrainingConfig, device_manager: WanDeviceManager):
    from diffusers import WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(
        config.model_id, subfolder="transformer", torch_dtype=config.torch_dtype(), low_cpu_mem_usage=True
    )
    transformer = device_manager.to_device(transformer)
    for p in transformer.parameters():
        p.requires_grad_(False)
    if config.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    transformer.add_adapter(_make_lora_config(config))
    device_manager.shard_model(transformer)

    total = sum(p.numel() for p in transformer.parameters())
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    assert trainable > 0, "no trainable LoRA params; check lora_targets"
    assert trainable < total // 20, "trainable params suspiciously large; LoRA not isolated"
    return transformer
