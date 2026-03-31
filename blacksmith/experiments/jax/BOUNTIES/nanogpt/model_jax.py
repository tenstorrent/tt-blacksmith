# SPDX-FileCopyrightText: (c) 2022 Andrej Karpathy
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import FlaxGPT2LMHeadModel


@dataclass(frozen=True)
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # This was originally 50257 for GPT-2.0, but padded to the minimal higher multiple of 64 for effeciency.
    )
    num_layers: int = 12
    num_heads: int = 12
    num_embeds: int = 768
    dropout_rate: float = 0.0
    use_bias: bool = True
    use_matmul_embed: bool = False
    dtype: Optional[str] = None


class MatMulEmbed(nn.Module):
    """
    A 'Fallback' Embedding that uses Matrix Multiplication (supported on TT)
    instead of Gather/Scatter (not supported on TT).
    """

    num_embeddings: int
    features: int
    dtype: any = jnp.float32  # <--- Use bf16 to save memory (50MB vs 100MB)

    @nn.compact
    def __call__(self, inputs):
        # 1. One-Hot Encode (Creates [B, T, V])
        # This consumes memory, but JAX/XLA optimizes the lifetime better than PyTorch.
        x = jax.nn.one_hot(inputs, self.num_embeddings, dtype=self.dtype)

        # 2. Define Weights [V, C]
        embedding = self.param(
            "embedding", jax.nn.initializers.normal(stddev=0.02), (self.num_embeddings, self.features), self.dtype
        )

        # 3. MatMul: [B, T, V] @ [V, C] -> [B, T, C]
        return jnp.dot(x, embedding)

    def attend(self, query):
        # For the Language Model Head (Weight Tying)
        embedding = self.variables["params"]["embedding"]
        return jnp.dot(query, embedding.T)


class CausalSelfAttention(nn.Module):

    num_heads: int
    dtype: Any = jnp.float32
    dropout_rate: float = 0.1
    deterministic: Optional[bool] = None
    use_proj_bias: bool = True

    @nn.compact
    def __call__(self, x, mask, deterministic=None):
        B, T, C = x.shape
        assert C % self.num_heads == 0
        head_dim = C // self.num_heads
        deterministic = nn.merge_param("deterministic", self.deterministic, deterministic)

        qkv = nn.Dense(3 * C, use_bias=self.use_proj_bias, dtype=self.dtype, name="c_attn")(x)
        qkv = qkv.reshape(B, T, 3 * self.num_heads, head_dim)
        q, k, v = jnp.array_split(qkv, 3, axis=2)
        # Calculating attention matrix.
        scale = 1.0 / jnp.sqrt(head_dim).astype(self.dtype)
        # Attention weight shape is (batch..., num_heads, q_length, kv_length).
        attn = jnp.einsum("...qhd,...khd->...hqk", q, k) * scale
        attn = attn + mask  # Add the causal mask.

        # We are doing manual softmax for to prevent over/underflows on tt hardware.
        attn_max = jnp.max(attn, axis=-1, keepdims=True)
        shifted_attn = attn - jax.lax.stop_gradient(attn_max)
        exp_attn = jnp.exp(shifted_attn)
        attn = exp_attn / jnp.sum(exp_attn, axis=-1, keepdims=True)

        attn = attn.astype(self.dtype)
        attn = nn.Dropout(self.dropout_rate)(attn, deterministic=deterministic)

        # Return weighted sum over values for each query position.
        x = jnp.einsum("...hqk,...khd->...qhd", attn, v).reshape(B, T, C)
        x = nn.Dense(C, use_bias=self.use_proj_bias, dtype=self.dtype, name="c_proj")(x)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x


class MLP(nn.Module):
    config: GPTConfig

    @nn.compact
    def __call__(self, x, deterministic=None):
        B, T, C = x.shape
        x = nn.Dense(4 * C, dtype=self.config.dtype, use_bias=self.config.use_bias, name="c_fc")(x)
        x = nn.gelu(x, approximate=True)  # We use approximization to avoid calculatin Gaussian Error Function (erf).
        x = nn.Dense(C, dtype=self.config.dtype, use_bias=self.config.use_bias, name="c_proj")(x)
        x = nn.Dropout(self.config.dropout_rate)(x, deterministic)
        return x


class Block(nn.Module):
    config: GPTConfig

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-4, dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.attn = CausalSelfAttention(self.config.num_heads, self.config.dtype, dropout_rate=self.config.dropout_rate)
        self.ln_2 = nn.LayerNorm(epsilon=1e-4, dtype=self.config.dtype, use_bias=self.config.use_bias)
        self.mlp = MLP(self.config)

    def __call__(self, x, mask=None, deterministic=None):
        x = x + self.attn(self.ln_1(x), mask, deterministic)
        x = x + self.mlp(self.ln_2(x), deterministic)
        return x


class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        # 1. Embeddings
        if self.config.use_matmul_embed:
            self.wte = MatMulEmbed(self.config.vocab_size, self.config.num_embeds, name="wte")
            self.wpe = MatMulEmbed(self.config.block_size, self.config.num_embeds, name="wpe")
        else:
            self.wte = nn.Embed(self.config.vocab_size, self.config.num_embeds, name="wte")
            self.wpe = nn.Embed(self.config.block_size, self.config.num_embeds, name="wpe")
        self.drop = nn.Dropout(self.config.dropout_rate)  # Removed for compiler safety

        # 2. Transformer Blocks
        self.blocks = [Block(self.config, name=str(i)) for i in range(self.config.num_layers)]

        # 3. Final LayerNorm
        self.ln_f = nn.LayerNorm(1e-4, dtype=self.config.dtype, use_bias=self.config.use_bias, name="ln_f")

        def init_pos_ids(rng):
            # Shape: [1, Block_Size]
            return jnp.array(np.arange(self.config.block_size, dtype=np.uint32)[None, :])

        def init_causal_mask(rng):
            # Shape: [1, 1, Block_Size, Block_Size]
            # Created via NumPy to avoid JAX boolean issues on TT backend.
            mask = np.tri(self.config.block_size, k=0, dtype=np.float32)
            mask_bias = (1.0 - mask) * -10000.0
            return jnp.array(mask_bias[None, None, :, :], dtype=jnp.float32)

        self.pos_ids = self.variable("cache", "pos_ids", init_pos_ids, None)
        self.mask = self.variable("cache", "mask", init_causal_mask, None)

    def __call__(self, idx, deterministic=None):
        B, T = idx.shape

        # Slice for current sequence length.
        pos_ids = self.pos_ids.value[:, :T]
        mask = self.mask.value[:, :, :T, :T]

        x = self.embed(idx, pos_ids, deterministic)
        x = self.body(x, mask, deterministic)
        logits = self.head(x)
        return logits

    def embed(self, idx, pos_ids, deterministic=None):
        token_embed = self.wte(idx)
        pos_embed = self.wpe(pos_ids)
        x = token_embed + pos_embed
        x = self.drop(x, deterministic=deterministic)  # Removed
        return x

    def body(self, x, mask, deterministic=None):
        B, T, C = x.shape
        # mask = nn.make_causal_mask(jnp.ones((B, T), dtype=jnp.int32), dtype=bool)
        # We crate the mask on CPU during runtime to avoid JAX boolean issues on TT backend.
        for block in self.blocks:
            x = block(x, mask, deterministic=deterministic)

        x = self.ln_f(x)
        return x

    def head(self, x):
        return self.wte.attend(x)

    def init(self, rng):
        tokens = jnp.zeros((2, self.config.block_size), dtype=jnp.uint32)
        params = jax.jit(super().init, static_argnums=(2,))(rng, tokens, True)
        return params


def convert_hf_params(hf_params: FrozenDict, num_heads, num_embeds) -> FrozenDict:
    params = unfreeze(hf_params)
    for k, v in params.pop("h", {}).items():
        params[k] = v

    params = flatten_dict(params, sep=".")
    for k in params.keys():
        # if k.endswith('attn.c_attn.bias'):
        #    params[k] = params[k].reshape(num_heads, -1)
        if k.endswith("attn.c_attn.kernel"):
            # params[k] = params[k].reshape(num_embeds, num_heads, -1)
            params[k] = params[k].T
        elif k.endswith("attn.c_proj.kernel"):
            # params[k] = params[k].reshape(num_heads, -1, num_embeds)
            params[k] = params[k].T
        elif k.split(".")[1] == "mlp" and k.endswith("kernel"):
            params[k] = params[k].T

    params = unflatten_dict({f"params.{k}": v for k, v in params.items()}, sep=".")
    return freeze(params)


def get_pretrained_params(model_type: str) -> Tuple[GPTConfig, FrozenDict]:
    """
    returns config and pretrained parameters from huggingface gpt models
    """
    assert model_type in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl")
    # Only dropout can be overridden, see more notes below.
    print("loading weights from pretrained gpt: %s" % model_type)

    config = {
        "gpt2": GPTConfig(num_layers=12, num_heads=12, num_embeds=768),  # 124M params
        "gpt2-medium": GPTConfig(num_layers=24, num_heads=16, num_embeds=1024),  # 350M params
        "gpt2-large": GPTConfig(num_layers=36, num_heads=20, num_embeds=1280),  # 774M params
        "gpt2-xl": GPTConfig(num_layers=48, num_heads=25, num_embeds=1600),  # 1.558B params
    }[model_type]

    model_hf = FlaxGPT2LMHeadModel.from_pretrained(model_type)
    hf_params = model_hf.params["transformer"]
    params = convert_hf_params(hf_params, config.num_heads, config.num_embeds)
    return config, params
