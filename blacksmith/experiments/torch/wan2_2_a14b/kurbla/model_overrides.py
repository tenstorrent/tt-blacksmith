# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


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


def apply_generality_overrides() -> None:
    _patch_umt5_relative_bias_dtensor()
    _patch_tp_conv_support_check()
