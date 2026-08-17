# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import time
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


def _patch_umt5_layer_norm_dtensor() -> None:
    # UmT5LayerNorm computes variance = hidden_states.pow(2).mean(-1).  When the
    # row-parallel `o`/`wo` linears shard the hidden-size dimension across the batch
    # mesh axis, that mean(-1) becomes a partial reduction and DTensor emits an
    # all_reduce avg collective.  The TT compile backend cannot lower that op.
    #
    # Fix: all-gather the hidden-size shards before the RMS computation so mean(-1) is
    # purely local.  The weight multiply that follows is unaffected because layer_norm
    # weights are already sharded on hidden-size (Shard(0) on batch), so DTensor sees a
    # consistent placement and emits no extra collectives.
    try:
        from transformers.models.umt5 import modeling_umt5 as umt5
    except ImportError:
        return

    from torch.distributed.tensor import DTensor, Replicate
    from torch.distributed.tensor.placement_types import Shard as DTShard

    _orig_forward = umt5.UMT5LayerNorm.forward

    def forward(self, hidden_states):
        if isinstance(hidden_states, DTensor):
            mesh = hidden_states.device_mesh
            # Replace any Shard placement on the hidden-size (last) dim with Replicate
            # so mean(-1) sees a full, local tensor.
            last_dim = hidden_states.dim() - 1
            new_placements = [
                Replicate() if isinstance(p, DTShard) and p.dim == last_dim else p
                for p in hidden_states.placements
            ]
            if new_placements != list(hidden_states.placements):
                hidden_states = hidden_states.redistribute(mesh, new_placements)
        return _orig_forward(self, hidden_states)

    umt5.UMT5LayerNorm.forward = forward


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

    # Signature mirrors upstream compute_bias (query_length, key_length, device, cache_position);
    # past_seen_tokens is kept for older transformers releases that pass an offset instead.
    def compute_bias(self, query_length, key_length, device=None, cache_position=None, past_seen_tokens=0):
        weight = self.relative_attention_bias.weight
        if device is None:
            device = weight.device
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None] + past_seen_tokens
        else:
            context_position = cache_position[:, None]
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


def _patch_timestep_embedding_dtensor() -> None:
    # Build the sinusoidal timestep embedding from the local timesteps, then re-wrap.
    #
    # get_timestep_embedding derives its frequency table from torch.arange, so it is a
    # plain tensor, and line 63 multiplies it by `timesteps` -- a DTensor once the DiT
    # is sharded, because generate.py hands the timestep over with to_device. DTensor
    # refuses mixed operands, and unlike the rope's freqs_cos/freqs_sin (registered
    # buffers, which distribute_module already replicated) this table is constructed
    # fresh on every call, so there is nothing for shard_model to have converted.
    #
    # The timestep is replicated, so every chip computes the identical embedding from
    # identical inputs: a local computation re-labelled Replicate, no collective. Same
    # shape as the UMT5 relative-bias patch above.
    try:
        from diffusers.models import embeddings as diffusers_embeddings
    except ImportError:
        return

    from torch.distributed.tensor import DTensor, Replicate

    _orig_get_timestep_embedding = diffusers_embeddings.get_timestep_embedding

    def get_timestep_embedding(timesteps, *args, **kwargs):
        if not isinstance(timesteps, DTensor):
            return _orig_get_timestep_embedding(timesteps, *args, **kwargs)
        mesh = timesteps.device_mesh
        replicated = [Replicate()] * mesh.ndim
        if not all(isinstance(p, Replicate) for p in timesteps.placements):
            # A sharded timestep would make each chip embed a different slice; gather
            # first so the local result is the whole answer.
            timesteps = timesteps.redistribute(mesh, replicated)
        emb = _orig_get_timestep_embedding(timesteps.to_local(), *args, **kwargs)
        return DTensor.from_local(emb, device_mesh=mesh, placements=replicated, run_check=False)

    diffusers_embeddings.get_timestep_embedding = get_timestep_embedding


def _patch_rms_norm_dtensor() -> None:
    # Recompute nn.RMSNorm out-of-place, with a sum instead of a mean.
    #
    # The DiT's attn.norm_q/norm_k normalize over the hidden dim, which is sharded on
    # "model" (["model"] in the YAML), and their input arrives from the column-parallel
    # to_q/to_k as Partial(sum) over "batch". torch.rms_norm's decomposition ends with an
    # in-place `add_(eps)` on the variance, and completing that variance needs a
    # collective, so DTensor refuses: "in-place operations that require placement changes
    # are not supported".
    #
    # Two things this does differently. It writes the same formula out-of-place, so the
    # redistribution DTensor already wanted is allowed to happen. And it reduces with sum
    # rather than mean: a reduction over the sharded last dim leaves Partial(sum), which
    # `+ eps` completes with an all_reduce(sum) of a single scalar per token, where mean
    # would leave Partial(avg) -- the collective the tt backend cannot lower (see
    # _patch_umt5_layer_norm_dtensor). The hidden shard is never gathered.
    from torch.distributed.tensor import DTensor, Replicate
    from torch.distributed.tensor.placement_types import Partial as DTPartial

    _orig_forward = nn.RMSNorm.forward

    def forward(self, x):
        if not isinstance(x, DTensor) or len(self.normalized_shape) != 1:
            return _orig_forward(self, x)

        # Resolve Partial before squaring: x currently holds each shard's contribution to
        # a sum, and the square of a partial sum is not the partial sum of squares. Doing
        # it here rather than leaving it to the pointwise op keeps the collective visible.
        if any(isinstance(p, DTPartial) for p in x.placements):
            x = x.redistribute(
                x.device_mesh,
                [Replicate() if isinstance(p, DTPartial) else p for p in x.placements],
            )

        dtype = x.dtype
        x_f32 = x.float()
        # .shape is the global shape, so this is the full normalized width even though
        # each chip holds a slice of it.
        width = x_f32.shape[-1]
        eps = self.eps if self.eps is not None else torch.finfo(torch.float32).eps
        sum_sq = x_f32.pow(2).sum(-1, keepdim=True)
        out = (x_f32 * torch.rsqrt(sum_sq / width + eps)).to(dtype)
        if self.weight is not None:
            out = out * self.weight
        return out

    nn.RMSNorm.forward = forward


def _patch_sdpa_partial_dtensor() -> None:
    # Materialize any Partial operand before attention.
    #
    # A Partial tensor holds one chip's contribution to a sum, so it is only valid input
    # to linear ops. Attention is not linear -- softmax(sum_i x_i) != sum_i softmax(x_i)
    # -- so q/k/v have to be summed first. The column-parallel to_q/to_k/to_v contract
    # over a "batch"-sharded input dim, which leaves all three Partial(sum) over "batch".
    # Wan's qk-norm resolves q and k on the way through (_patch_rms_norm_dtensor), but
    # nothing touches v between to_v and the attention call.
    #
    # Left alone, DTensor reconciles the mismatched operands itself and picks a
    # reduce-scatter of the sequence dim, which tt-kurbla implements as a collective
    # needing an even split -- and 390 tokens do not divide by 4:
    #   ValueError: tt_kurbla.reduce_scatter: dim 2 (size 390) is not divisible by 4
    # Resolving Partial -> Replicate here is the all-reduce the arithmetic required
    # anyway, and it leaves the head shard over "model" untouched, so attention still
    # runs distributed over heads.
    import torch.nn.functional as F
    from torch.distributed.tensor import DTensor, Replicate
    from torch.distributed.tensor.placement_types import Partial as DTPartial

    _orig_sdpa = F.scaled_dot_product_attention

    def _materialize(tensor):
        if isinstance(tensor, DTensor) and any(isinstance(p, DTPartial) for p in tensor.placements):
            return tensor.redistribute(
                tensor.device_mesh,
                [Replicate() if isinstance(p, DTPartial) else p for p in tensor.placements],
            )
        return tensor

    def scaled_dot_product_attention(query, key, value, *args, **kwargs):
        return _orig_sdpa(_materialize(query), _materialize(key), _materialize(value), *args, **kwargs)

    F.scaled_dot_product_attention = scaled_dot_product_attention


def _patch_conv3d_out_channel_sharded_weight() -> None:
    # Compute a Conv3d locally when its weight is sharded on out-channels.
    #
    # DTensor has no convolution sharding rule for that layout: with
    # `_tp_conv._is_supported` forced to True (WanDeviceManager._patch_dtensor_conv) it
    # copies the input's spec onto the output, so the DiT's patch_embedding
    # (["batch", null, null, null, null] -> local (768, 48, 1, 2, 2) of a 3072-channel
    # weight) returns a Replicate() result whose global shape still says 3072 channels
    # while the local tensor holds 768. The next `hidden_states.flatten(2)` then asks
    # for [1, 3072, 390] from 299520 local elements and raises.
    #
    # With a replicated input, an out-channel shard of the weight produces exactly that
    # chip's slice of the output channels -- no data from other shards is involved -- so
    # the local convolution is the whole answer and Shard(1) is its true placement. That
    # is also the placement the column-parallel blocks downstream expect, so this emits
    # no collective at all. Anything else (sharded input, weight sharded on another dim,
    # bias not sharded to match) falls through to the stock forward.
    from torch.distributed.tensor import DTensor, Replicate
    from torch.distributed.tensor.placement_types import Shard as DTShard

    _orig_forward = nn.Conv3d.forward

    def forward(self, input):
        weight = self.weight
        if not isinstance(weight, DTensor):
            return _orig_forward(self, input)

        shards = [(mesh_dim, p.dim) for mesh_dim, p in enumerate(weight.placements) if isinstance(p, DTShard)]
        if len(shards) != 1 or shards[0][1] != 0:
            return _orig_forward(self, input)
        mesh_dim = shards[0][0]

        if isinstance(input, DTensor) and not all(isinstance(p, Replicate) for p in input.placements):
            return _orig_forward(self, input)

        bias = self.bias
        if isinstance(bias, DTensor):
            # The bias must be split the same way, or the local conv would add the full
            # bias vector to a channel slice.
            bias_shards = [(md, p.dim) for md, p in enumerate(bias.placements) if isinstance(p, DTShard)]
            if bias_shards != [(mesh_dim, 0)]:
                return _orig_forward(self, input)
            bias = bias.to_local()
        elif bias is not None:
            return _orig_forward(self, input)

        mesh = weight.device_mesh
        local_input = input.to_local() if isinstance(input, DTensor) else input
        local_out = self._conv_forward(local_input, weight.to_local(), bias)
        placements = [DTShard(1) if d == mesh_dim else Replicate() for d in range(mesh.ndim)]
        return DTensor.from_local(local_out, device_mesh=mesh, placements=placements, run_check=False)

    nn.Conv3d.forward = forward


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
    _patch_umt5_layer_norm_dtensor()
    _patch_umt5_relative_bias_dtensor()
    _patch_timestep_embedding_dtensor()
    # _patch_rms_norm_dtensor()
    # _patch_sdpa_partial_dtensor()
    # _patch_conv3d_out_channel_sharded_weight()


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

    # Phase timings: "why is a run slow" is otherwise guesswork between a 10 GB checkpoint
    # read, the per-parameter device move, and sharding thousands of tensors.
    t0 = time.perf_counter()
    transformer = WanTransformer3DModel.from_pretrained(
        config.model_id, subfolder="transformer", torch_dtype=config.torch_dtype(), low_cpu_mem_usage=True
    )
    print(f"[build] DiT checkpoint loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    if config.dit_layers is not None and config.dit_layers < len(transformer.blocks):
        # Drop the tail blocks before anything moves to device or gets sharded, so the
        # run pays for neither. Slicing an nn.ModuleList returns a new ModuleList, and
        # the forward iterates `self.blocks`, so nothing else has to change. The model's
        # own config still reports 30 layers -- a checkpoint saved from a truncated run
        # would be wrong, which is why this is a bring-up knob only.
        print(f"[dit] truncating to the first {config.dit_layers} of {len(transformer.blocks)} blocks")
        transformer.blocks = transformer.blocks[: config.dit_layers]
    # Backend-specific per-instance rewrites (a no-op on tt-xla); must precede the move
    # so the device only ever sees modules the backend can lower.
    t0 = time.perf_counter()
    transformer = device_manager.prepare_model(transformer)
    transformer = device_manager.to_device(transformer)
    print(f"[build] DiT moved to device in {time.perf_counter() - t0:.1f}s", flush=True)
    for p in transformer.parameters():
        p.requires_grad_(False)
    if config.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    t0 = time.perf_counter()
    transformer.add_adapter(_make_lora_config(config))
    device_manager.shard_model(transformer)
    print(f"[build] LoRA + sharding in {time.perf_counter() - t0:.1f}s", flush=True)

    total = sum(p.numel() for p in transformer.parameters())
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    assert trainable > 0, "no trainable LoRA params; check lora_targets"
    assert trainable < total // 20, "trainable params suspiciously large; LoRA not isolated"
    return transformer
