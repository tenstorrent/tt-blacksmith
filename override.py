"""Tenstorrent-side overrides + device management for `wan22_5b.py`.

Mirrors the structure of `tt-xla:ppadjin/wan5b_tests` (see WAN5B_TT_NOTES.md):

- `apply_generality_overrides()` — correctness-only patches (the model will
  not run on TT/XLA without these). Apply once at import time.
- `apply_perf_overrides()`       — graph-break removals + sharding-friendly
  rewrites. Apply once at import time, before any `torch.compile` call.
- `enable_spmd()`                — sets the env vars + flips xr SPMD mode.
- `wan22_mesh()`                 — 2D ("batch", "model") SPMD mesh sized to
                                   the current device count.
- shard-spec builders            — per-component dict[Tensor, partition_spec]
                                   ready to feed `xs.mark_sharding`.
- `WanDeviceManager`             — single entry point that holds the mesh,
                                   moves modules to the XLA device, applies
                                   shard specs, and compiles for backend "tt"
                                   the same way the branch's `run_component`
                                   does.

Specific override bodies (the actual monkey patches) will be filled in once
the user picks which ones to port from the branch.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# SPMD setup (mirror of tests/infra/utilities/torch_multichip_utils.py)
# ---------------------------------------------------------------------------


def enable_spmd() -> None:
    """Enable torch_xla SPMD mode. Idempotent; cannot be disabled once set.

    `CONVERT_SHLO_TO_SHARDY=1` makes the pytorch-xla fork emit Shardy
    annotations into the StableHLO it hands to tt-mlir (the tt-mlir
    stablehlo pipeline expects them).
    """
    import torch_xla.runtime as xr

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


# Mirrors `shared.wan22_mesh`. Adapts to the available device count so the
# same script works on a 1-device dev box and an N-device llmbox.
_MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}


def wan22_mesh():
    """2D ("batch", "model") SPMD mesh sized to current device count."""
    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh

    n = xr.global_runtime_device_count()
    if n not in _MESH_SHAPES:
        raise ValueError(
            f"Unsupported device count: {n}. Expected one of {sorted(_MESH_SHAPES)}."
        )
    mesh_shape = _MESH_SHAPES[n]
    device_ids = np.array(range(n))
    return Mesh(device_ids, mesh_shape, ("batch", "model"))


# ---------------------------------------------------------------------------
# Shard specs (mirrors of shared.shard_*_specs)
# ---------------------------------------------------------------------------


def shard_umt5_specs(encoder) -> dict:
    """UMT5EncoderModel shard specs.

    Mesh axes: ("batch", "model")
    Column-parallel (q, k, v, wi_0, wi_1):  ("model", "batch")
    Row-parallel   (o, wo):                 ("batch", "model")
    """
    specs = {encoder.shared.weight: (None, "batch")}

    for block in encoder.encoder.block:
        sa = block.layer[0].SelfAttention
        specs[sa.q.weight] = ("model", "batch")
        specs[sa.k.weight] = ("model", "batch")
        specs[sa.v.weight] = ("model", "batch")
        specs[sa.o.weight] = ("batch", "model")
        specs[block.layer[0].layer_norm.weight] = ("batch",)

        ffn = block.layer[1].DenseReluDense
        specs[ffn.wi_0.weight] = ("model", "batch")
        specs[ffn.wi_1.weight] = ("model", "batch")
        specs[ffn.wo.weight] = ("batch", "model")
        specs[block.layer[1].layer_norm.weight] = ("batch",)

    specs[encoder.encoder.final_layer_norm.weight] = ("batch",)
    return specs


def shard_vae_encoder_specs(vae) -> dict:
    """AutoencoderKLWan encoder shard specs (seeds sharding through 3D Causal VAE)."""
    return {
        vae.quant_conv.weight: ("batch", None, None, None, None),
        vae.quant_conv.bias: ("batch",),
        vae.encoder.conv_in.weight: ("batch", None, None, None, None),
        vae.encoder.conv_in.bias: ("batch",),
    }


def shard_vae_decoder_specs(vae) -> dict:
    """AutoencoderKLWan decoder shard specs (mirror of encoder)."""
    return {
        vae.post_quant_conv.weight: ("batch", None, None, None, None),
        vae.post_quant_conv.bias: ("batch",),
        vae.decoder.conv_in.weight: ("batch", None, None, None, None),
        vae.decoder.conv_in.bias: ("batch",),
    }


def _dit_linear_weight(layer: nn.Module) -> torch.Tensor:
    """Base weight for a DiT linear; LoRA-wrapped layers expose `base_layer`."""
    return getattr(layer, "base_layer", layer).weight


def _dit_linear_bias(layer: nn.Module) -> torch.Tensor:
    return getattr(layer, "base_layer", layer).bias


def _lora_ab_weights(layer: nn.Module):
    """`(lora_A.weight, lora_B.weight)` for a PEFT-wrapped linear, else `(None, None)`.

    PEFT stores adapters in `lora_A` / `lora_B` ModuleDicts keyed by adapter
    name. `lora_A.weight` is `(r, in_features)`; `lora_B.weight` is
    `(out_features, r)`. Layers that were not LoRA-adapted (no `lora_A`)
    return `(None, None)`.
    """
    a = getattr(layer, "lora_A", None)
    b = getattr(layer, "lora_B", None)
    if a is None or b is None:
        return None, None
    a_mod = a["default"] if "default" in a else next(iter(a.values()))
    b_mod = b["default"] if "default" in b else next(iter(b.values()))
    return a_mod.weight, b_mod.weight


def shard_dit_specs(dit) -> dict:
    """WanTransformer3DModel shard specs.

    Intended to run after LoRA adapters are applied. Frozen base weights AND
    the trainable LoRA A/B weights are sharded.

    Mesh axes: ("batch", "model")
    Column-parallel (QKV, FFN up):  ("model", "batch")
    Row-parallel   (O, FFN down):   ("batch", "model")

    LoRA A/B are sharded to match the layout their gradients already receive
    from the sharded base matmuls, so the fused AdamW step stays element-wise
    (no replicated-param pull -> no resharding -> no `sdy.collective_permute`,
    which tt-mlir cannot lower; see tt-mlir#3370). The rank dim is always
    replicated:
        col W=("model","batch") -> lora_A=(None,"batch"), lora_B=("model",None)
        row W=("batch","model") -> lora_A=(None,"model"), lora_B=("batch",None)
    """
    specs = {
        dit.patch_embedding.weight: ("batch", None, None, None, None),
        dit.patch_embedding.bias: ("batch",),
        dit.scale_shift_table: (None, None, "batch"),
        dit.proj_out.weight: (None, "batch"),
        dit.proj_out.bias: (None,),
    }

    ce = dit.condition_embedder
    specs[ce.time_embedder.linear_1.weight] = ("model", "batch")
    specs[ce.time_embedder.linear_1.bias] = ("model",)
    specs[ce.time_embedder.linear_2.weight] = ("batch", "model")
    specs[ce.time_embedder.linear_2.bias] = ("batch",)
    specs[ce.time_proj.weight] = ("batch", None)
    specs[ce.time_proj.bias] = ("batch",)
    specs[ce.text_embedder.linear_1.weight] = ("model", "batch")
    specs[ce.text_embedder.linear_1.bias] = ("model",)
    specs[ce.text_embedder.linear_2.weight] = ("batch", "model")
    specs[ce.text_embedder.linear_2.bias] = ("batch",)

    # parallel: "col" (QKV / FFN-up) or "row" (O / FFN-down). No-op if the
    # layer wasn't LoRA-adapted.
    def _add_lora(layer: nn.Module, parallel: str) -> None:
        a_w, b_w = _lora_ab_weights(layer)
        if a_w is None:
            return
        if parallel == "col":
            specs[a_w] = (None, "batch")
            specs[b_w] = ("model", None)
        else:
            specs[a_w] = (None, "model")
            specs[b_w] = ("batch", None)

    for block in dit.blocks:
        specs[block.scale_shift_table] = (None, None, "batch")
        specs[block.norm2.weight] = ("batch",)
        specs[block.norm2.bias] = ("batch",)

        for attn in [block.attn1, block.attn2]:
            specs[_dit_linear_weight(attn.to_q)] = ("model", "batch")
            specs[_dit_linear_bias(attn.to_q)] = ("model",)
            specs[_dit_linear_weight(attn.to_k)] = ("model", "batch")
            specs[_dit_linear_bias(attn.to_k)] = ("model",)
            specs[_dit_linear_weight(attn.to_v)] = ("model", "batch")
            specs[_dit_linear_bias(attn.to_v)] = ("model",)
            specs[_dit_linear_weight(attn.to_out[0])] = ("batch", "model")
            specs[_dit_linear_bias(attn.to_out[0])] = ("batch",)
            specs[attn.norm_q.weight] = ("model",)
            specs[attn.norm_k.weight] = ("model",)

            _add_lora(attn.to_q, "col")
            _add_lora(attn.to_k, "col")
            _add_lora(attn.to_v, "col")
            _add_lora(attn.to_out[0], "row")

        ffn_up = block.ffn.net[0].proj
        ffn_down = block.ffn.net[2]
        specs[_dit_linear_weight(ffn_up)] = ("model", "batch")
        specs[_dit_linear_bias(ffn_up)] = ("model",)
        specs[_dit_linear_weight(ffn_down)] = ("batch", "model")
        specs[_dit_linear_bias(ffn_down)] = ("batch",)

        _add_lora(ffn_up, "col")
        _add_lora(ffn_down, "row")

    return specs


# ---------------------------------------------------------------------------
# Monkey patches — bodies to be filled in once the user picks which to apply.
# Each helper is split into a "generality" bucket (correctness; needed for the
# code to run at all on TT/XLA) and a "perf" bucket (graph-break removals,
# SPMD-friendly rewrites). All patches are no-ops until populated.
# ---------------------------------------------------------------------------


# ----- Generality patches --------------------------------------------------


def _patch_wan_resample_rep_sentinel() -> None:
    """[generality] Replace the `"Rep"` string sentinel in
    `WanResample.forward` with `object()` + `is` / `is not`.

    Reason: `feat_cache[idx] == "Rep"` triggers `Tensor.__eq__(str)` once
    the slot holds a tensor, which dynamo cannot trace -> graph break.

    Note: subsumed by `_patch_wan_resample_avoid_4d_fold` (which bakes the
    same fix in). Either alone is sufficient; applying both is harmless.
    """
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
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] is not rep
                    ):
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] is rep
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )
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
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                    )
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    akw.WanResample.forward = forward


_ORIG_GETITEM = torch.Tensor.__getitem__


def _clamp_slice(s: slice, size: int) -> slice:
    """Canonicalize a slice into ``[0, size]`` (positive step) or ``[-1, size-1]``
    (negative step), like ``slice.indices(size)`` would.

    Hand-written instead of ``slice(...).indices(size)`` because the latter is
    a CPython slot wrapper and dynamo cannot symbolically execute it — it
    graph-breaks at trace time. Plain ``max``/``min``/comparisons on concrete
    ints are fully traceable.
    """
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
            remaining_explicit = sum(
                x is not Ellipsis and x is not None for x in idx[idx.index(item) + 1 :]
            )
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

        # Leave tensor / bool / advanced indices untouched.
        out.append(item)
        dim += 1

    return tuple(out)


class _SafeSlicingMode(torch.overrides.TorchFunctionMode):
    """Intercept ``Tensor.__getitem__`` via a stack-managed function mode.

    A slot-reassignment approach (``torch.Tensor.__getitem__ = ...``) looks
    reversible but permanently flips a CPython "this type has Python
    overrides" flag, which disables PyTorch's fast path inside
    ``torch.tensor(list_of_tensors)`` — the fallback then calls ``__len__`` on
    each 0-d element and raises. The diffusers UniPC scheduler hits exactly
    that (``b = torch.tensor(b, device=device)``) after the first VAE decode.
    A ``TorchFunctionMode`` is the supported, properly-reversible mechanism.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.Tensor.__getitem__:
            self_, idx = args
            return _ORIG_GETITEM(self_, _normalize_index(idx, self_.shape))
        return func(*args, **kwargs)


@contextmanager
def safe_xla_slicing():
    """[generality] Wrap a region with a `TorchFunctionMode` that clamps
    out-of-range slice indices in `Tensor.__getitem__` before re-dispatch.

    Reason: CPU silently clamps slice `start`/`stop` outside `[-size, size]`;
    torch-xla raises "Value out of range". AutoencoderKLWan relies on the
    CPU behavior (e.g. `x[:, :, -2:, :, :]` on a size-1 temporal dim).

    Must wrap the entire run (CPU golden + dynamo trace + TT execute).
    """
    with _SafeSlicingMode():
        yield


# ----- Perf patches (graph-break removals; no correctness impact) ---------


def _patch_apply_lora_scale() -> None:
    """[perf] Make `diffusers.utils.peft_utils.apply_lora_scale` a no-op.

    Reason: the decorator wraps DiT `forward` in `scale_lora_layers` /
    `unscale_lora_layers`, each a graph break. Also rebind
    `WanTransformer3DModel.forward` to `forward.__wrapped__` because the
    decorator is applied at class-definition time, so patching the helper
    alone doesn't undo the already-wrapped class attribute.

    Note: for the LoRA training script the LoRA adapters themselves are
    real; this patch only kills the diffusers _helper_ overhead.
    """
    from diffusers.utils import peft_utils

    def noop_decorator(kwargs_name: str = "joint_attention_kwargs"):
        def decorator(forward_fn):
            return forward_fn

        return decorator

    peft_utils.apply_lora_scale = noop_decorator

    # WanTransformer3DModel.forward is decorated at class-definition time, so
    # patching the helper alone doesn't undo the already-wrapped attribute.
    from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

    wrapped = WanTransformer3DModel.forward
    underlying = getattr(wrapped, "__wrapped__", None)
    if underlying is not None:
        WanTransformer3DModel.forward = underlying


def _patch_wan_time_embedder_dtype() -> None:
    """[perf] Replace `.parameters()` dtype probe in `WanTimeTextImageEmbedding`.

    Upstream uses `next(iter(self.time_embedder.parameters())).dtype` to cast
    the sinusoidal timestep features to the embedder weight dtype before
    `TimestepEmbedding.forward`. Under dynamo that `.parameters()` call
    graph-breaks and the resume sub-graph hits
    `NameError: cannot access free variable 'named_children'`.
    `linear_1` is always the first parameter of `TimestepEmbedding` — same
    dtype, direct attribute access traces cleanly.
    """
    import torch
    from diffusers.models.transformers.transformer_wan import WanTimeTextImageEmbedding

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
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
    """[perf] Pop `tt_torch.torch_overrides.torch_function_override` off the
    `TorchFunctionMode` stack.

    Reason: `tt_torch.torch_overrides` enters a `TorchFunctionMode` at
    import time. Its body is gated on `torch.compiler.is_compiling()` and
    is a no-op on the compile path, but the mode still sits on dynamo's
    function-mode stack and forces a `__torch_function__` trace on every
    matmul / linear encountered while tracing.
    """
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
        # Mode wasn't on the stack or was already popped — ignore.
        pass


def _patch_wan_resample_avoid_4d_fold() -> None:
    """[generality] Replace the 5D->4D permute+reshape + nn.Upsample path in
    `WanResample.forward` with per-slice unbind / repeat_interleave /
    Conv2d / stack so dim-1 channel sharding survives SPMD.

    Reason (two correctness regressions stack into ~0.40 PCC on a sharded
    `up_blocks[2]` at 480p, ~0.9 on the full sharded decoder):
      1. The original `permute(0,2,1,3,4).reshape(b*t, c, h, w)` loses
         dim-1 channel sharding through the partitioner.
      2. `WanUpsample` (nn.Upsample, `mode="nearest-exact"`) lowers to a
         tt-mlir kernel that produces wrong values on channel-sharded
         inputs.

    Per-slice unbind + repeat_interleave on H/W + Conv2d + stack is
    SPMD-clean and bit-equivalent to nearest-exact at 2x scale. Without
    this the sharded decoder runs but outputs are wrong, so this is a
    correctness fix, not a perf optimization (despite involving rewrite of
    a hot op).

    Also bakes in the rep-sentinel fix from
    `_patch_wan_resample_rep_sentinel` (so it supersedes that patch).

    Critically, the per-slice unbind/stack also removes the dynamo graph
    breaks the original `nn.Upsample` + `== "Rep"` path triggers, so the
    whole per-frame `_decode` loop traces into a single compiled graph
    instead of one graph per frame threading `feat_cache` in/out.
    """
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
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] is not rep
                    ):
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] is rep
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if feat_cache[idx] is rep:
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)

        # Spatial resample. Per-slice unbind T -> manual 2x upsample ->
        # Conv2d -> stack T. SPMD-clean for any T and break-free for dynamo.
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
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                    )
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    akw.WanResample.forward = forward


# ----- Public entry points -------------------------------------------------


def apply_generality_overrides() -> None:
    """Apply all generality-bucket patches. Call once at import time, before
    model load. `safe_xla_slicing()` still needs to be entered as a context
    manager around the actual run.

    `_patch_wan_resample_avoid_4d_fold` supersedes
    `_patch_wan_resample_rep_sentinel` (it copies the rep-sentinel fix
    inline), so applying both is harmless but the avoid-4d-fold call alone
    is sufficient.
    """
    _patch_wan_resample_rep_sentinel()
    _patch_wan_resample_avoid_4d_fold()


def apply_perf_overrides() -> None:
    """Apply all performance-bucket patches (graph-break removals only;
    these change speed, not output values). Call once at import time,
    before any `torch.compile` call.
    """
    _patch_apply_lora_scale()
    _patch_wan_time_embedder_dtype()
    _disable_tt_torch_function_override()


# ---------------------------------------------------------------------------
# Device manager (mirror of shared.run_component, but exposed as a class so
# the training loop can hold one instance for the lifetime of a run)
# ---------------------------------------------------------------------------


# Component name -> shard-spec builder. The builder takes the **inner**
# diffusers module (encoder, vae, dit), not the wrapper, so the LoRA-wrapped
# transformer can be sharded by passing `transformer` here even if it's
# wrapped in a `*Wrapper`.
SHARD_BUILDERS: dict[str, Callable[[nn.Module], dict]] = {
    "umt5": shard_umt5_specs,
    "vae_encoder": shard_vae_encoder_specs,
    "vae_decoder": shard_vae_decoder_specs,
    "dit": shard_dit_specs,
}


class WanDeviceManager:
    """Hold an SPMD mesh + cached `torch.compile` wrappers for a Wan 2.2 run.

    Mirrors the way `tests/torch/models/wan5b/shared.run_component` manages
    a component on the XLA device:

    1. `enable_spmd()` once if any sharded module exists.
    2. `xr.set_device_type("TT")` + `xm.xla_device()` for the device handle.
    3. `set_custom_compile_options(...)` from a `CompilerConfig`-equivalent.
    4. Move module + inputs to device.
    5. `xs.mark_sharding(tensor, mesh, spec)` for every (tensor, spec) in
       `SHARD_BUILDERS[component](inner_module)`.
    6. Cache `torch.compile(..., backend="tt")` on `(id(module), shape_sig)`
       so repeat calls don't re-trace.

    Caller invariant: keep wrappers alive across calls. If a wrapper is
    GC'd and a new one created, `id()` can be reused for a *different*
    object — a silent cache hit returning the wrong compiled graph.
    """

    def __init__(
        self,
        use_tt: bool = True,
        sharded: bool = True,
        xla_compile_options: Optional[dict] = None,
        torch_compile_options: Optional[dict] = None,
    ) -> None:
        self.use_tt = use_tt
        self.sharded = sharded
        self._xla_compile_options = (
            xla_compile_options
            if xla_compile_options is not None
            else self._default_xla_compile_options()
        )
        self._torch_compile_options = (
            torch_compile_options
            if torch_compile_options is not None
            else self._default_torch_compile_options()
        )
        self.mesh = None
        self.device = None
        self._compile_cache: dict = {}

        if use_tt:
            self._setup_tt()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _default_xla_compile_options() -> dict:
        """Options passed to `torch_xla.set_custom_compile_options(...)`.

        Values must be strings — the torch_xla compile-options API does
        not coerce Python `bool` / `int`.
        """
        return {
            "optimization_level": "0",
            "fp32_dest_acc_en": "true",
            "math_fidelity": "hifi4",
            # Match ppadjin/wan5b_tests CompilerConfig (shared.run_component).
            "experimental-enable-dram-space-saving-optimization": "true",
        }

    @staticmethod
    def _default_torch_compile_options() -> dict:
        """Options passed to `torch.compile(..., backend="tt", options=...)`.

        These are TT backend knobs interpreted by the dynamo backend, not
        XLA — they accept Python types.
        """
        return {
            "tt_enable_torch_fx_fusion_pass": False,
            "tt_legacy_compile": True,
        }

    def _setup_tt(self) -> None:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        if self.sharded:
            enable_spmd()

        xr.set_device_type("TT")
        os.environ.setdefault("PJRT_DEVICE", "TT")
        os.environ.setdefault("XLA_STABLEHLO_COMPILE", "1")

        self.device = xm.xla_device()
        torch_xla.set_custom_compile_options(self._xla_compile_options)

        if self.sharded:
            self.mesh = wan22_mesh()
            print(f"[device] mesh shape={self.mesh.mesh_shape} axes={self.mesh.axis_names}")

    # -- module + tensor movement --------------------------------------------

    def to_device(self, module_or_tensor):
        """Move a module or tensor to the run device."""
        return module_or_tensor.to(self.device)

    def shard_module(self, inner_module: nn.Module, component: str) -> None:
        """Apply `xs.mark_sharding` to every parameter the matching shard
        builder returns. `inner_module` must be the actual diffusers module
        (e.g. `vae` or `transformer`), not a `*Wrapper`.

        Re-applying `mark_sharding` with identical specs on already-sharded
        tensors is a no-op in torch_xla, so this is safe to call multiple
        times.
        """
        if not (self.use_tt and self.sharded and self.mesh is not None):
            return
        if len(self.mesh.device_ids) <= 1:
            return

        import torch_xla.distributed.spmd as xs

        builder = SHARD_BUILDERS.get(component)
        if builder is None:
            raise KeyError(
                f"Unknown component {component!r}; expected one of {sorted(SHARD_BUILDERS)}"
            )
        specs = builder(inner_module)
        for tensor, spec in specs.items():
            xs.mark_sharding(tensor, self.mesh, spec)

    # -- compile + run --------------------------------------------------------

    def compile(self, module: nn.Module):
        """Wrap `module` in `torch.compile(backend="tt", options=...)` (TT
        path) or pass it through (CPU/CUDA path). Result is cached on
        `id(module)` only — callers should pass a stable, long-lived module
        (re-creating a wrapper after `del` can reuse the same id and return
        the wrong compiled graph).
        """
        if not self.use_tt:
            return module
        cached = self._compile_cache.get(id(module))
        if cached is None:
            cached = torch.compile(
                module, backend="tt", options=self._torch_compile_options
            )
            self._compile_cache[id(module)] = cached
        return cached

    def sync(self) -> None:
        """Block until pending XLA ops have completed. No-op off TT."""
        if not self.use_tt:
            return
        import torch_xla

        torch_xla.sync(wait=True)
