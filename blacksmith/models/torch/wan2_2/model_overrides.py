# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import contextmanager

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from blacksmith.experiments.torch.wan2_2.configs import TrainingConfig

if TYPE_CHECKING:
    # Type-only: the concrete manager depends on the backend (`models/torch/wan2_2/device.py`
    # for tt-xla, `experiments/torch/wan2_2/kurbla/device_manager.py` for tt-kurbla). Importing
    # it eagerly would drag `torch_xla` into the pure-torch path.
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

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
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


def _clamp_bound(value, size: int, none_default: int, step_positive: bool) -> int:
    # Clamp one slice endpoint the way slice.indices(size) would. For step > 0 the
    # valid range is [0, size]; for step < 0 it is [-1, size - 1]. `none_default` is
    # the value slice.indices uses when the endpoint is omitted.
    if value is None:
        return none_default
    lo, hi = (0, size) if step_positive else (-1, size - 1)
    if value < 0:
        return max(lo, value + size)
    return min(value, hi)


def _clamp_slice(s: slice, size: int) -> slice:
    # Canonicalize a slice into bounds the way slice.indices(size) would, but with
    # plain int math (slice.indices is a slot wrapper dynamo graph-breaks on).
    step = 1 if s.step is None else s.step
    pos = step > 0
    start = _clamp_bound(s.start, size, 0 if pos else size - 1, pos)
    stop = _clamp_bound(s.stop, size, size if pos else -1, pos)
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
            self_, idx = args[0], args[1]
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

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
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


def _patch_wan_avgdown_avoid_8d_permute() -> None:
    # Replace AvgDown3D's 8D view+permute with per-sub-lattice slicing + stack.
    #
    # The original views [B,C,T,H,W] as [B,C,T//ft,ft,H//fs,fs,W//fs,fs] before
    # permuting, which parks the size-`fs` factor axis in the last dimension. Tile
    # layout pads the trailing dim to a multiple of 32, so an fs=2 axis pads 2->32
    # and the buffer is 16x the logical tensor: the first encoder down block asks
    # for 511 MB to hold a 32 MB activation, and the device OOMs.
    #
    # Slicing the (t,h,w) sub-lattices instead keeps (H', W') as the trailing dims
    # of every intermediate. Bit-equivalent: the original's view splits index
    # `t = t'*ft + t_off`, which is exactly `x[:, :, t_off::ft]`, and the folded
    # channel index is `c*factor + (t_off*fs*fs + h_off*fs + w_off)` either way.
    try:
        from diffusers.models.autoencoders import autoencoder_kl_wan as akw
    except ImportError:
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ft, fs = self.factor_t, self.factor_s
        pad_t = (ft - x.shape[2] % ft) % ft
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        parts = [
            x[:, :, t_off::ft, h_off::fs, w_off::fs]
            for t_off in range(ft)
            for h_off in range(fs)
            for w_off in range(fs)
        ]
        x = torch.stack(parts, dim=2)
        x = x.reshape(b, c * self.factor, t // ft, h // fs, w // fs)
        x = x.reshape(b, self.out_channels, self.group_size, t // ft, h // fs, w // fs)
        return x.mean(dim=2)

    akw.AvgDown3D.forward = forward


def _patch_umt5_relative_bias_dtensor() -> None:
    # Promote UMT5Attention's relative-position bucket to a replicated DTensor before
    # the embedding lookup.
    #
    # compute_bias derives the bucket from torch.arange, so it is a plain tensor, while
    # relative_attention_bias.weight is a DTensor once the encoder is sharded. DTensor
    # refuses mixed operands ("got mixed torch.Tensor and DTensor"), and this is the only
    # place in the encoder where a locally-constructed index tensor meets a distributed
    # weight. The bucket is identical on every device, so Replicate() on each mesh dim is
    # the correct placement and the lookup needs no collective.
    try:
        from transformers.models.umt5 import modeling_umt5 as umt5
    except ImportError:
        return

    from torch.distributed.tensor import DTensor, Replicate

    def compute_bias(self, query_length, key_length, device=None, past_seen_tokens=0):
        weight = self.relative_attention_bias.weight
        if device is None:
            device = weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(relative_position)
        if isinstance(weight, DTensor) and not isinstance(relative_position_bucket, DTensor):
            mesh = weight.device_mesh
            relative_position_bucket = DTensor.from_local(
                relative_position_bucket,
                device_mesh=mesh,
                placements=[Replicate()] * mesh.ndim,
                run_check=False,
            )
        values = self.relative_attention_bias(relative_position_bucket)  # (q_len, k_len, num_heads)
        return values.permute([2, 0, 1]).unsqueeze(0)  # (1, num_heads, q_len, k_len)

    umt5.UMT5Attention.compute_bias = compute_bias


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
    _patch_wan_avgdown_avoid_8d_permute()
    _patch_umt5_relative_bias_dtensor()


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
        # 
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


def build_lora_transformer(config: TrainingConfig, device_manager: "WanDeviceManager"):
    from diffusers import WanTransformer3DModel

    transformer = WanTransformer3DModel.from_pretrained(
        config.model_id, subfolder="transformer", torch_dtype=config.torch_dtype(), low_cpu_mem_usage=True
    )
    # Backend-specific per-instance rewrites (a no-op on tt-xla); must precede the move
    # so the device only ever sees modules the backend can lower.
    transformer = device_manager.prepare_model(transformer)
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
