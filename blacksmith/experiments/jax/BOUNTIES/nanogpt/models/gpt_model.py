# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
from typing import Optional, Tuple, Any
import math
import logging

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer."""
    
    n_embd: int
    n_head: int
    dropout: float = 0.0
    bias: bool = False
    
    def setup(self):
        # Key, query, value projections for all heads
        self.c_attn = nn.Dense(3 * self.n_embd, use_bias=self.bias)
        # Output projection
        self.c_proj = nn.Dense(self.n_embd, use_bias=self.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values for all heads in batch
        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)
        
        # Reshape to separate heads
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(k.shape[-1]))
        
        # Create causal mask
        mask = jnp.tril(jnp.ones((T, T)))
        att = jnp.where(mask == 0, float('-inf'), att)
        
        att = nn.softmax(att, axis=-1)
        att = self.attn_dropout(att, deterministic=not training)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)  # (B, T, nh, hs) -> (B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y), deterministic=not training)
        return y


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""
    
    n_embd: int
    dropout: float = 0.0
    bias: bool = False
    
    def setup(self):
        self.c_fc = nn.Dense(4 * self.n_embd, use_bias=self.bias)
        self.c_proj = nn.Dense(self.n_embd, use_bias=self.bias)
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = self.c_fc(x)
        x = nn.gelu(x)
        x = self.c_proj(x)
        x = self.dropout_layer(x, deterministic=not training)
        return x


class Block(nn.Module):
    """Transformer block with self-attention and MLP."""
    
    n_embd: int
    n_head: int
    dropout: float = 0.0
    bias: bool = False
    
    def setup(self):
        self.ln_1 = nn.LayerNorm(use_bias=self.bias)
        self.attn = CausalSelfAttention(
            n_embd=self.n_embd,
            n_head=self.n_head,
            dropout=self.dropout,
            bias=self.bias
        )
        self.ln_2 = nn.LayerNorm(use_bias=self.bias)
        self.mlp = MLP(
            n_embd=self.n_embd,
            dropout=self.dropout,
            bias=self.bias
        )
        
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # Self-attention with residual connection
        x = x + self.attn(self.ln_1(x), training=training)
        # MLP with residual connection
        x = x + self.mlp(self.ln_2(x), training=training)
        return x


class GPT(nn.Module):
    """GPT model implementation in Flax."""
    
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = False
    
    def setup(self):
        # Token and position embeddings
        self.wte = nn.Embed(self.vocab_size, self.n_embd)
        self.wpe = nn.Embed(self.block_size, self.n_embd)
        self.drop = nn.Dropout(self.dropout)
        
        # Transformer blocks
        self.blocks = [Block(
            n_embd=self.n_embd,
            n_head=self.n_head,
            dropout=self.dropout,
            bias=self.bias
        ) for _ in range(self.n_layer)]
        
        # Final layer norm and language modeling head
        self.ln_f = nn.LayerNorm(use_bias=self.bias)
        self.lm_head = nn.Dense(self.vocab_size, use_bias=False)
        
    def __call__(self, idx: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        device = idx.device if hasattr(idx, 'device') else None
        B, T = idx.shape
        
        # Token embeddings
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        
        # Position embeddings
        pos = jnp.arange(0, T, dtype=jnp.int32)
        pos_emb = self.wpe(pos)  # (T, n_embd)
        
        # Combine embeddings
        x = self.drop(tok_emb + pos_emb, deterministic=not training)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, training=training)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits
    
    def get_num_params(self, params: Any) -> int:
        """Get the number of parameters in the model."""
        def count_params(p):
            if isinstance(p, dict):
                return sum(count_params(v) for v in p.values())
            elif hasattr(p, 'shape'):
                return p.size
            else:
                return 0
        
        return count_params(unfreeze(params))


def create_model(config) -> GPT:
    """Create a GPT model from configuration."""
    logger.info(f"Creating GPT model with config: {config}")
    model = GPT(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias
    )
    logger.info("GPT model created successfully")
    return model


def estimate_mfu(model: GPT, fwdbwd_per_iter: int, dt: float) -> float:
    """Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS."""
    # First estimate the number of flops we do per iteration.
    # See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    
    # Number of parameters
    N = model.get_num_params(model.params) if hasattr(model, 'params') else 0
    
    cfg = model.config if hasattr(model, 'config') else None
    if cfg is None:
        return 0.0
    
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    
    # Express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0/dt) # per second
    flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu
