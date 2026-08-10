# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


# class Conv3dAsMatmul(nn.Conv3d):

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         b, c, t, h, w = x.shape
#         kt, kh, kw = self.kernel_size
#         # Enforced here rather than at install time so a config change surfaces loudly.
#         assert self.stride == self.kernel_size, f"Conv3dAsMatmul needs stride == kernel_size, got {self.stride} vs {self.kernel_size}"
#         assert tuple(self.padding) == (0, 0, 0), f"Conv3dAsMatmul needs zero padding, got {self.padding}"
#         assert tuple(self.dilation) == (1, 1, 1), f"Conv3dAsMatmul needs dilation 1, got {self.dilation}"
#         assert self.groups == 1, f"Conv3dAsMatmul needs groups == 1, got {self.groups}"
#         assert t % kt == 0 and h % kh == 0 and w % kw == 0, (
#             f"input {(t, h, w)} is not divisible by kernel {(kt, kh, kw)}; "
#             "resolutions must be a multiple of VAE_stride * patch_size"
#         )

#         ot, oh, ow = t // kt, h // kh, w // kw
#         # (B, C, ot, kt, oh, kh, ow, kw) -> (B, ot, oh, ow, C, kt, kh, kw): output
#         # positions first (in conv output order), then the patch contents in weight order.
#         patches = x.view(b, c, ot, kt, oh, kh, ow, kw).permute(0, 2, 4, 6, 1, 3, 5, 7)
#         patches = patches.reshape(b, ot * oh * ow, c * kt * kh * kw)
#         out = torch.nn.functional.linear(patches, self.weight.reshape(self.out_channels, -1), self.bias)
#         # Back to conv's (B, out_channels, ot, oh, ow) so the caller's reshape/flatten
#         # (WanTransformer3DModel.forward does `.flatten(2).transpose(1, 2)`) is unchanged.
#         return out.transpose(1, 2).reshape(b, self.out_channels, ot, oh, ow)


# class CausalConv3dAsConv2dStack(nn.Conv3d):

#     def forward(self, x: torch.Tensor, cache_x: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # `_padding` is (w, w, h, h, 2 * pad_t, 0) -- see WanCausalConv3d.__init__.
#         pad_w, _, pad_h, _, pad_t, pad_t_after = self._padding
#         assert pad_t_after == 0, f"causal conv expects no trailing temporal pad, got {pad_t_after}"

#         if cache_x is not None and pad_t > 0:
#             cache_x = cache_x.to(x.device)
#             x = torch.cat([cache_x, x], dim=2)
#             pad_t -= cache_x.shape[2]
#         if pad_t > 0:
#             # cat of zeros instead of F.pad: aten.constant_pad_nd is not lowered.
#             zeros = x.new_zeros(x.shape[0], x.shape[1], pad_t, x.shape[3], x.shape[4])
#             x = torch.cat([zeros, x], dim=2)

#         kernel_t = self.kernel_size[0]
#         stride_t = self.stride[0]
#         batch, _, frames = x.shape[0], x.shape[1], x.shape[2]
#         frames_out = (frames - kernel_t) // stride_t + 1
#         assert frames_out > 0, f"input has {frames} frames, too few for a {kernel_t}-tap temporal kernel"

#         out = None
#         for k in range(kernel_t):
#             # Time-slice this tap reads, then fold time into the batch dim for conv2d.
#             taps = x[:, :, k : k + (frames_out - 1) * stride_t + 1 : stride_t]
#             taps = taps.permute(0, 2, 1, 3, 4).reshape(batch * frames_out, x.shape[1], x.shape[3], x.shape[4])
#             # Bias is added once, on the first tap, not once per tap.
#             partial = torch.nn.functional.conv2d(
#                 taps,
#                 self.weight[:, :, k],
#                 self.bias if (k == 0 and self.bias is not None) else None,
#                 stride=self.stride[1:],
#                 padding=(pad_h, pad_w),
#                 dilation=self.dilation[1:],
#                 groups=self.groups,
#             )
#             partial = partial.reshape(batch, frames_out, *partial.shape[1:]).permute(0, 2, 1, 3, 4)
#             out = partial if out is None else out + partial
#         return out


def zero_pad_via_cat(x: torch.Tensor, pad: Sequence[int]) -> torch.Tensor:
    assert len(pad) % 2 == 0, f"pad must be pairs of (before, after), got {len(pad)} entries"
    for i in range(len(pad) // 2):
        before, after = pad[2 * i], pad[2 * i + 1]
        if not before and not after:
            continue
        dim = x.dim() - 1 - i
        blocks = []
        if before:
            shape = list(x.shape)
            shape[dim] = before
            blocks.append(x.new_zeros(shape))
        blocks.append(x)
        if after:
            shape = list(x.shape)
            shape[dim] = after
            blocks.append(x.new_zeros(shape))
        x = torch.cat(blocks, dim=dim)
    return x


class ZeroPad2dAsCat(nn.ZeroPad2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.ZeroPad2d.padding is (left, right, top, bottom) == F.pad order.
        return zero_pad_via_cat(x, self.padding)


def rewrite_zero_pad2d_as_cat(model: nn.Module) -> int:
    """Retype every `nn.ZeroPad2d` in `model` to `ZeroPad2dAsCat`."""
    count = 0
    for module in model.modules():
        if type(module) is nn.ZeroPad2d:
            module.__class__ = ZeroPad2dAsCat
            count += 1
    return count


def rewrite_causal_conv3d_as_conv2d(model: nn.Module) -> int:
    """Retype every `WanCausalConv3d` in `model` to `CausalConv3dAsConv2dStack`.

    Matched by class name so this module does not have to import the VAE internals
    (and so it no-ops on a model that has none, e.g. the DiT).
    """
    count = 0
    for module in model.modules():
        if type(module).__name__ == "WanCausalConv3d":
            module.__class__ = CausalConv3dAsConv2dStack
            count += 1
    return count


def rewrite_conv3d_as_matmul(model: nn.Module) -> int:
    """Retype every non-overlapping `nn.Conv3d` in `model` to `Conv3dAsMatmul`.

    Returns the number of modules rewritten. Overlapping/padded Conv3d (the Wan VAE's
    3x3x3 `WanCausalConv3d`) is left alone -- the matmul form does not apply and those
    need real conv3d support in the backend.
    """
    count = 0
    for module in model.modules():
        if type(module) is nn.Conv3d and module.stride == module.kernel_size and tuple(module.padding) == (0, 0, 0):
            module.__class__ = Conv3dAsMatmul
            count += 1
    return count


def patch_wan_rope_without_scatter() -> None:
    from diffusers.models.transformers import transformer_wan as twan

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = twan._get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # `unflatten` is a view, and under data parallelism the BACKWARD of the value
        # unflatten receives a gradient in SDPA's internal (B, H, S, D) layout, which cannot
        # be viewed back to (B, S, H*D) -- "Cannot view a tensor with shape [4, 32, 24, 128]
        # and strides (98304, 128, 4096, 1)". `reshape` copies when strides forbid a view,
        # in both directions. Forward is unaffected, so this only appears once the batch is
        # sharded.
        query = query.reshape(*query.shape[:2], attn.heads, -1)
        key = key.reshape(*key.shape[:2], attn.heads, -1)
        value = value.reshape(*value.shape[:2], attn.heads, -1)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                even = x1 * cos - x2 * sin
                odd = x1 * sin + x2 * cos
                # Interleave even/odd back into the last dim: (..., n, 2) -> (..., 2n).
                out = torch.stack((even, odd), dim=-1).flatten(-2)
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = twan._get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = twan.dispatch_attention_fn(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False,
                backend=self._attention_backend,
            )
            hidden_states_img = hidden_states_img.reshape(*hidden_states_img.shape[:2], -1)
            hidden_states_img = hidden_states_img.type_as(query)

        # Call SDPA in its native (B, H, S, D) layout with explicit, contiguous transposes
        # rather than through `dispatch_attention_fn`, which takes (B, S, H, D) and transposes
        # internally. Under data parallelism the internal version hands a gradient back in
        # (B, H, S, D) layout to a head-split that expects (B, S, H, D), and the merge back to
        # (B, S, H*D) is then not expressible as a view:
        #   "Cannot view a tensor with shape [4, 32, 24, 128] and strides
        #    (98304, 128, 4096, 1) as a tensor with shape (4, 32, 3072)"
        # Owning the transposes keeps every reshape on a contiguous tensor, so both the
        # forward and its backward stay expressible. Costs one copy per attention.
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = hidden_states.reshape(*hidden_states.shape[:2], -1)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    twan.WanAttnProcessor.__call__ = __call__


def patch_wan_avg_down3d_pad() -> None:
    """Rebuild `AvgDown3D.forward` with `zero_pad_via_cat` instead of `F.pad`.

    `AvgDown3D` temporally zero-pads up to a multiple of `factor_t`
    (`autoencoder_kl_wan.py:58`), which silently returns all zeros on tt. It is a
    functional `F.pad` call, not a module, so it cannot be fixed by retyping -- the
    whole forward is rebound, copied verbatim from diffusers 0.35.1 apart from the pad.
    """
    from diffusers.models.autoencoders import autoencoder_kl_wan as akw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = zero_pad_via_cat(x, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x.shape
        x = x.view(
            B, C, T // self.factor_t, self.factor_t,
            H // self.factor_s, self.factor_s, W // self.factor_s, self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.view(
            B, self.out_channels, self.group_size,
            T // self.factor_t, H // self.factor_s, W // self.factor_s,
        )
        return x.mean(dim=2)

    akw.AvgDown3D.forward = forward


def patch_timestep_embedding_for_dtensor() -> None:
    """Make `get_timestep_embedding` accept a DTensor timestep.

    diffusers builds the sinusoid frequencies as a fresh *plain* tensor and then does
    `timesteps[:, None].float() * emb[None, :]` (`diffusers/models/embeddings.py`).
    On a mesh the timestep arrives as a replicated DTensor, and DTensor refuses mixed
    operands: "aten.mul.Tensor got mixed torch.Tensor and DTensor, need to convert all
    torch.Tensor to DTensor before calling distributed operators!".

    The embedding is a pure elementwise function of its input, so for a *replicated*
    timestep the mesh-correct answer is just the local computation re-wrapped as
    replicated -- no collective, no placement choice to get wrong. A sharded timestep
    would need real thought, so this refuses it rather than silently doing the wrong
    math (see the CONTEXT.md trap: a wrong-but-plausible tensor still scores a good pcc).

    Delete this when diffusers builds the frequency tensor on/for the input's mesh, or
    when tt-kurbla grows DTensor-aware constant materialization.
    """
    import diffusers.models.embeddings as _embeddings
    from torch.distributed.tensor import DTensor, Replicate

    if getattr(_embeddings.get_timestep_embedding, "_kurbla_dtensor_safe", False):
        return

    _original = _embeddings.get_timestep_embedding

    def get_timestep_embedding(timesteps, *args, **kwargs):
        if isinstance(timesteps, DTensor):
            mesh = timesteps.device_mesh
            # (B,) -> (B, dim): elementwise along the batch dim, so a batch shard carries
            # through unchanged and a replicated input stays replicated. Anything else (a
            # shard of some other dim) has no meaning for a 1-D timestep, so refuse it.
            placements = []
            for placement in timesteps.placements:
                if placement.is_replicate():
                    placements.append(Replicate())
                elif placement.is_shard() and placement.dim == 0:
                    placements.append(placement)
                else:
                    raise NotImplementedError(
                        "kurbla: get_timestep_embedding handles a replicated or batch-sharded "
                        f"timestep, got placements {timesteps.placements}"
                    )
            local = _original(timesteps.to_local(), *args, **kwargs)
            return DTensor.from_local(local, mesh, placements, run_check=False)
        return _original(timesteps, *args, **kwargs)

    get_timestep_embedding._kurbla_dtensor_safe = True
    _embeddings.get_timestep_embedding = get_timestep_embedding

    # The shared overrides re-export the symbol; rebind any module that grabbed it by value.
    import blacksmith.models.torch.wan2_2.model_overrides as _shared

    if hasattr(_shared, "get_timestep_embedding"):
        _shared.get_timestep_embedding = get_timestep_embedding


def register_fused_backward_sharding() -> None:
    """Give DTensor sharding rules for tt-kurbla's fused backward ops.

    tt-kurbla lowers `aten::linear` / `aten::matmul` and their backwards as single ops
    rather than decomposing to mm + t + sum (tt-kurbla 16d7003). DTensor has strategies
    for the decomposed ops but has never heard of the fused backwards, so the first LoRA
    training step on a mesh dies with:

        NotImplementedError: Operator aten.linear_backward.default does not have a
        sharding strategy registered.

    Schemas:
        linear_backward(self, grad_output, weight, output_mask) -> (grad_self, grad_weight, grad_bias)
        matmul_backward(grad, self, other, mask)                -> (grad_self, grad_other)

    These register the **replicated** case only, which is the rung the mesh ladder is on:
    all-replicated operands give all-replicated gradients and no collective is required.
    Sharded operands need real placement algebra (a data-parallel grad_weight is Partial
    over the batch axis and must be all-reduced), so anything non-replicated is refused
    loudly rather than silently producing a plausible-looking wrong gradient -- per the
    CONTEXT.md trap, wrong-but-finite numbers still score a good pcc.

    Delete this when tt-kurbla either decomposes these backwards or ships its own
    DTensor strategies.
    """
    import torch
    from torch.distributed.tensor import Replicate
    from torch.distributed.tensor.experimental import register_sharding

    if getattr(register_fused_backward_sharding, "_done", False):
        return

    def _require_replicated(op: str, **specs) -> None:
        for name, spec in specs.items():
            if not all(p.is_replicate() for p in spec.placements):
                raise NotImplementedError(
                    f"kurbla: {op} on a mesh only supports replicated operands; {name} has "
                    f"placements {spec.placements}. Sharded operands need a Partial gradient "
                    "plus an all-reduce, which is not implemented."
                )

    def _fused_backward_strategies(n_tensor_inputs, output_mask, weight_index=2):
        """Acceptable single-mesh-dim strategies for a fused backward op.

        `register_sharding` describes strategies for ONE mesh axis; DTensor expands them
        across the full mesh, choosing per axis. So rather than reasoning about the 4x8 mesh
        as a whole, list the two layouts that are meaningful on any single axis:

        1. **everything replicated** -> replicated gradients, no collective.
        2. **data parallel**: activations sharded on the batch dim, weight replicated.
           `grad_input` stays batch-sharded; `grad_weight`/`grad_bias` are a sum over the
           batch and so come back `Partial()`. DTensor lowers that to an all-reduce when the
           gradient is redistributed onto the replicated parameter -- i.e. exactly the
           gradient all-reduce that data parallelism requires, for free.

        Tensor parallelism (a weight sharded along the contraction dim) is NOT listed: it
        needs its own algebra, and leaving it out means DTensor reports "no strategy" rather
        than silently picking a wrong-but-plausible one.
        """
        from torch.distributed.tensor import Partial, Shard

        def outputs(grad_in, grad_w):
            # Mirror `output_mask`, except that a bias-free Linear with a trainable weight
            # (mask (T,T,F)) still gets a materialized grad_bias back from the kernel; a
            # None spec for a real tensor trips "output tensor should be scalar!" in
            # DTensor's wrap(). That combination is exactly a LoRA A/B projection.
            produced = list(output_mask)
            if len(produced) == 3 and produced[1] and not produced[2]:
                produced[2] = True
            grads = [grad_in, grad_w, grad_w][: len(produced)]
            return [g if keep else None for g, keep in zip(grads, produced)]

        replicated = (outputs(Replicate(), Replicate()), [Replicate()] * n_tensor_inputs)
        data_parallel = (
            outputs(Shard(0), Partial()),
            # self/grad_output batch-sharded, weight replicated, non-tensor args None.
            [Shard(0) if i < weight_index else Replicate() for i in range(n_tensor_inputs)],
        )
        return [replicated, data_parallel]

    @register_sharding(torch.ops.aten.linear_backward.default)
    def _linear_backward(self, grad_output, weight, output_mask):
        import os as _os
        if _os.environ.get("KURBLA_DEBUG_SHARDING"):
            print(f"DBG linear_backward mask={output_mask} "
                  f"self={self.placements} go={grad_output.placements} w={weight.placements}", flush=True)
        # linear_backward(self, grad_output, weight, output_mask)
        #   -> (grad_self, grad_weight, grad_bias)
        # The bool[] mask is a static arg: DTensor does not include it in args_schema,
        # so the input spec list covers only the three tensors.
        return _fused_backward_strategies(3, output_mask, weight_index=2)

    @register_sharding(torch.ops.aten.matmul_backward.default)
    def _matmul_backward(grad, self, other, mask):
        import os as _os
        if _os.environ.get("KURBLA_DEBUG_SHARDING"):
            print(f"DBG matmul_backward mask={mask} grad={grad.placements} "
                  f"self={self.placements} other={other.placements}", flush=True)
        """matmul_backward(grad, self, other, mask) -> (grad_self, grad_other)

        NOT the same shape as linear_backward. In a linear, `weight` is a replicated
        parameter, so its gradient is a sum over the batch and comes back `Partial()`. In a
        bare matmul inside attention (q@k^T, attn@v) BOTH operands are activations and are
        batch-sharded together, so both gradients are simply batch-sharded too -- there is
        nothing to reduce. Declaring `Partial()` here instead is wrong twice over: it asks
        for an all-reduce that must not happen, and it makes the local shapes disagree, which
        surfaces downstream as
        `INTERNAL ASSERT FAILED ... build_sum_to: reduction did not reach target shape`.
        """
        from torch.distributed.tensor import Shard

        produced = list(mask)
        return [
            ([Replicate() if k else None for k in produced], [Replicate()] * 3),
            ([Shard(0) if k else None for k in produced], [Shard(0)] * 3),
        ]

    # The flow-matching objective is an MSE, and tt-kurbla lowers mse_loss as a single op
    # too, so the loss itself needs a rule before the first backward can run.
    #   mse_loss(self, target, reduction) -> Tensor
    #   mse_loss_backward(grad_output, self, target, reduction) -> Tensor
    # Replicated operands reduce to the same scalar on every chip, so the result is
    # replicated and no collective is needed.
    from torch.distributed.tensor import Partial, Shard

    @register_sharding(torch.ops.aten.mse_loss.default)
    def _mse_loss(self, target, reduction=1):
        # Replicated inputs -> the same scalar everywhere. Batch-sharded inputs -> each chip
        # holds a partial reduction, so the loss is Partial and DTensor all-reduces it when
        # the scalar is read. (Mean-vs-sum: DTensor's Partial defaults to sum, so a sharded
        # `mean` reduction is only exact when every shard has the same element count, which
        # is guaranteed here because DTensor requires an even split.)
        return [
            ([Replicate()], [Replicate(), Replicate()]),
            ([Partial()], [Shard(0), Shard(0)]),
        ]

    @register_sharding(torch.ops.aten.mse_loss_backward.default)
    def _mse_loss_backward(grad_output, self, target, reduction):
        return [
            ([Replicate()], [Replicate(), Replicate(), Replicate()]),
            ([Shard(0)], [Replicate(), Shard(0), Shard(0)]),
        ]

    register_fused_backward_sharding._done = True


def apply_kurbla_overrides(model: nn.Module) -> dict:
    """Apply every tt-kurbla workaround to a constructed model; returns what it did.

    Class-level patches are idempotent, so this is safe to call more than once.
    """
    patch_wan_rope_without_scatter()
    patch_wan_avg_down3d_pad()
    patch_timestep_embedding_for_dtensor()
    register_fused_backward_sharding()
    return {
        "conv3d_as_matmul": rewrite_conv3d_as_matmul(model),
        "causal_conv3d_as_conv2d": rewrite_causal_conv3d_as_conv2d(model),
        "zero_pad2d_as_cat": rewrite_zero_pad2d_as_cat(model),
        "rope_without_scatter": True,
        "timestep_embedding_dtensor_safe": True,
        "fused_backward_sharding": True,
    }
