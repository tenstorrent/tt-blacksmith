# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn as nn


def _patch_umt5_relative_bias_dtensor() -> None:
    try:
        from transformers.models.umt5 import modeling_umt5 as umt5
    except ImportError:
        return

    from torch.distributed.tensor import DTensor, Replicate

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


def _patch_tp_conv_support_check() -> None:
    # WanCausalConv3d pads itself, so every conv reaches _tp_conv with padding 0.
    import torch.distributed.tensor._tp_conv as tp_conv

    def _is_supported(input_size, kernel_size, stride, padding, dilation):
        return True

    tp_conv._is_supported = _is_supported


def _patch_timestep_embedding_dtensor() -> None:
    # The arange frequency table is built per call, so shard_model never saw it.
    try:
        from diffusers.models import embeddings as diffusers_embeddings
    except ImportError:
        return

    from torch.distributed.tensor import DTensor

    _orig = diffusers_embeddings.get_timestep_embedding

    def get_timestep_embedding(timesteps, *args, **kwargs):
        if not isinstance(timesteps, DTensor):
            return _orig(timesteps, *args, **kwargs)
        mesh, placements = timesteps.device_mesh, timesteps.placements
        emb = _orig(timesteps.to_local(), *args, **kwargs)
        return DTensor.from_local(emb, device_mesh=mesh, placements=placements, run_check=False)

    diffusers_embeddings.get_timestep_embedding = get_timestep_embedding


def _patch_rms_norm_dtensor() -> None:
    # torch.rms_norm ends in an in-place add_(eps) on a Partial variance, which DTensor
    # refuses. Summed rather than meaned so the reduction completes as Partial(sum).
    _orig_forward = nn.RMSNorm.forward

    def forward(self, x):
        from torch.distributed.tensor import DTensor, Replicate
        from torch.distributed.tensor.placement_types import Partial

        if not isinstance(x, DTensor) or len(self.normalized_shape) != 1:
            return _orig_forward(self, x)

        # The square of a partial sum is not the partial sum of squares.
        if any(isinstance(p, Partial) for p in x.placements):
            x = x.redistribute(
                x.device_mesh, [Replicate() if isinstance(p, Partial) else p for p in x.placements]
            )

        dtype = x.dtype
        x_f32 = x.float()
        width = x_f32.shape[-1]  # global width, not this chip's slice
        eps = self.eps if self.eps is not None else torch.finfo(torch.float32).eps
        # Unsqueeze after the reduction: tt collectives mis-tile a size-1 last dim.
        sum_sq = x_f32.pow(2).sum(-1)
        scale = torch.rsqrt(sum_sq / width + eps)
        out = (x_f32 * scale.unsqueeze(-1)).to(dtype)
        return out * self.weight if self.weight is not None else out

    nn.RMSNorm.forward = forward


def _patch_wan_rope_out_of_place() -> None:
    # Eager tt silently drops the strided writes diffusers uses to fill the rotary
    # output, zeroing q and k. stack+flatten is bit-identical on CPU.
    try:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers import transformer_wan as wan
    except ImportError:
        return

    def rope(hidden_states, freqs_cos, freqs_sin):
        x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
        cos, sin = freqs_cos[..., 0::2], freqs_sin[..., 1::2]
        even = x1 * cos - x2 * sin
        odd = x1 * sin + x2 * cos
        return torch.stack((even, odd), dim=-1).flatten(-2).type_as(hidden_states)

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None, **kwargs
    ):
        assert attn.add_k_proj is None, "I2V path not ported; T2V-A14B has added_kv_proj_dim=null"
        query, key, value = wan._get_qkv_projections(attn, hidden_states, encoder_hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = rope(query, *rotary_emb)
            key = rope(key, *rotary_emb)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](hidden_states))

    wan.WanAttnProcessor.__call__ = __call__


def apply_generality_overrides() -> None:
    _patch_umt5_relative_bias_dtensor()
    _patch_tp_conv_support_check()
    _patch_timestep_embedding_dtensor()
    _patch_rms_norm_dtensor()
    _patch_wan_rope_out_of_place()
