# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging
import typing as tp

import easydel.modules
import jax
from easydel.layers.attention_operator._attention_impl import AttentionOutput
from easydel.layers.attention_operator.modules.vanilla import VanillaAttn
from eformer.escale import with_sharding_constraint
from flax.nnx.nn.dtypes import promote_dtype
from jax import Array
from jax import numpy as jnp

logger = logging.getLogger(__name__)


@jax.named_scope("easydel-vanillaimpl-native-xla-4d-repeat-kv")
def _vanilla_attn_forward_4d_repeat_kv(
    self,
    q: Array,
    k: Array,
    v: Array,
    mask: Array | None = None,
    bias: Array | None = None,
    init_bias: tp.Callable[[], Array] | None = None,
    deterministic: bool = True,
    dropout_rng: jax.random.PRNGKey = None,
    softmax_aux: Array | None = None,
    **ignore,
) -> AttentionOutput:
    """Drop-in replacement for easydel.layers...VanillaAttn.forward_native.

    EasyDeL's default GQA path reshapes Q to 5D (b, seq, kv_heads, num_reps, d)
    which tt-mlir cannot currently lower. This variant repeats K and V along
    the head axis so every tensor stays 4D throughout the attention
    computation, while preserving the original forward_native contract
    (including the lazy init_bias callback used by EasyDeL for causal masks).
    """
    sm_scale = self.metadata.softmax_scale
    sm_scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
    dtype = self.metadata.runtime_dtype
    softmax_dtype = self.metadata.runtime_softmax_dtype

    if softmax_dtype is None:
        softmax_dtype = jnp.float32

    model_mode = self.get_mode(q=q, BTHD=True)
    q_sharding, k_sharding, v_sharding, b_sharding, m_sharding, a_sharding = self.metadata.get_shardings(model_mode)
    with self.metadata.mesh:
        if bias is None and mask is None and init_bias is not None:
            bias = init_bias()

        b, qs, qh, d = q.shape
        b, ks, kh, d = k.shape
        num_reps = qh // kh

        q = with_sharding_constraint(arr=q, sharding=q_sharding)
        k = with_sharding_constraint(arr=k, sharding=k_sharding)
        v = with_sharding_constraint(arr=v, sharding=v_sharding)

        bias = with_sharding_constraint(arr=bias, sharding=b_sharding) if bias is not None else bias
        mask = with_sharding_constraint(arr=mask, sharding=m_sharding) if mask is not None else mask

        if num_reps > 1:
            k = jnp.repeat(k, num_reps, axis=2)
            v = jnp.repeat(v, num_reps, axis=2)

        q, k, v = promote_dtype((q, k, v), dtype=dtype)

        aw = jnp.einsum("bshd,bmhd->bhsm", q * sm_scale, k, optimize=True)

    if bias is not None:
        if bias.shape[1] == qh:
            pass
        elif bias.shape[1] == kh:
            bias = jnp.repeat(bias, num_reps, axis=1)
        elif bias.shape[1] == 1:
            bias = jnp.broadcast_to(bias, (b, qh, qs, ks))
        else:
            raise NotImplementedError("bias heads wont match!")
        aw = jnp.add(aw, bias.astype(aw.dtype))

    elif mask is not None:
        if mask.dtype != jnp.bool_:
            mask = mask.astype(jnp.bool_)

        if mask.ndim == 4:
            if mask.shape[1] == 1:
                mask = jnp.broadcast_to(mask, (b, 1, qs, ks))
            elif mask.shape[1] == kh:
                mask = jnp.repeat(mask, num_reps, axis=1)
            elif mask.shape[1] == qh:
                pass
            else:
                mask = jnp.broadcast_to(mask[:, :1], (b, 1, qs, ks))
        elif mask.ndim == 3:
            mask = jnp.reshape(mask, (b, 1, qs, ks))
        elif mask.ndim == 2:
            mask = jnp.reshape(mask, (b, 1, 1, ks))
            mask = jnp.broadcast_to(mask, (b, 1, qs, ks))
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        aw = jnp.where(mask, aw, jnp.finfo(aw.dtype).min)

    if softmax_aux is not None:
        if softmax_aux.ndim == 2:
            sinks = softmax_aux.reshape(1, kh, -1, 1)
            sinks = jnp.repeat(sinks, num_reps, axis=1)
            sinks = jnp.broadcast_to(sinks, (b, qh, qs, 1))
        elif softmax_aux.ndim == 1:
            sinks = softmax_aux.reshape(1, kh, -1, 1)
            sinks = jnp.repeat(sinks, num_reps, axis=1)
            sinks = jnp.broadcast_to(sinks, (b, qh, qs, 1))
        else:
            raise ValueError(f"Unsupported softmax_aux shape: {softmax_aux.shape}")
        combined_logits = jnp.concatenate([aw, sinks], axis=-1)
        combined_logits = combined_logits - jnp.max(combined_logits, axis=-1, keepdims=True)
        probs = jax.nn.softmax(combined_logits.astype(softmax_dtype), axis=-1).astype(dtype)
        aw = probs[..., :-1]
    else:
        aw = jax.nn.softmax(aw.astype(softmax_dtype), axis=-1).astype(dtype)

    dp = self.metadata.dropout_prob
    if not deterministic and dp > 0.0 and dropout_rng is not None:
        keep_prob = 1.0 - dp
        dropout_shape = tuple([1] * (k.ndim - 2)) + aw.shape[-2:]
        keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        aw = aw * multiplier

    attention = jnp.einsum("bhsm,bmhd->bshd", aw, v, optimize=True)

    return AttentionOutput(
        attention_weights=aw,
        attention_outputs=with_sharding_constraint(arr=attention, sharding=a_sharding),
    )


def apply_gqa_workaround() -> None:
    """Patch VanillaAttn.forward_native with the 4D repeat-KV variant.

    See _vanilla_attn_forward_4d_repeat_kv for rationale.
    """
    VanillaAttn.forward_native = _vanilla_attn_forward_4d_repeat_kv
    logger.info("Applied GQA 4D workaround: VanillaAttn.forward_native patched to avoid 5D tensors")
