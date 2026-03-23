# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

# ── Architecture constants (GPT OSS 20B) ─────────────────────────────
N_LAYERS = 24
N_EXPERTS = 32  # 20B uses 32 experts; 128 experts (80B) won't fit on 1×4 mesh
N_TOP_K = 4  # top-k routing
HIDDEN = 2880
INTER = 2880  # per-expert intermediate dim
N_HEADS = 64
N_KV_HEADS = 8
HEAD_DIM = 64  # per config: not hidden/n_heads
Q_DIM = N_HEADS * HEAD_DIM  # 4096
KV_DIM = N_KV_HEADS * HEAD_DIM  # 512
N_KV_GROUPS = N_HEADS // N_KV_HEADS  # 8
VOCAB_SIZE = 201088
MOE_ALPHA = 1.702
MOE_LIMIT = 7.0
RMS_EPS = 1e-5

# ── LoRA defaults ─────────────────────────────────────────────────────
LORA_RANK = 16
LORA_ALPHA_DEFAULT = 32
LORA_START_LAY = N_LAYERS // 2  # layers 12-23 get LoRA adapters on q_proj + v_proj


# ── Sharding helper (copied from tt_torch.sharding) ──────────────────
_MESH_IDX_PREFIX = "mesh_idx_"


def _partition_spec_to_sdy_sharding(mesh, partition_spec, unreduced=None) -> str:
    dim_shardings = []
    for axis in partition_spec:
        if axis is None:
            dim_shardings.append("{}")
        elif isinstance(axis, str):
            try:
                axis_idx = mesh.axis_names.index(axis)
                if mesh.mesh_shape[axis_idx] > 1:
                    dim_shardings.append(f'{{"{_MESH_IDX_PREFIX}{axis_idx}"}}')
                else:
                    dim_shardings.append("{}")
            except ValueError:
                dim_shardings.append("{}")
        elif isinstance(axis, (list, tuple)):
            axis_refs = []
            for ax_name in axis:
                if isinstance(ax_name, str):
                    try:
                        axis_idx = mesh.axis_names.index(ax_name)
                        axis_refs.append(f'"{_MESH_IDX_PREFIX}{axis_idx}"')
                    except ValueError:
                        pass
                elif isinstance(ax_name, int):
                    axis_refs.append(f'"{_MESH_IDX_PREFIX}{ax_name}"')
            dim_shardings.append("{" + ", ".join(axis_refs) + "}" if axis_refs else "{}")
        else:
            dim_shardings.append("{}")

    dims_str = ", ".join(dim_shardings)

    unreduced_str = ""
    if unreduced:
        unreduced_refs = []
        for ax in unreduced:
            if isinstance(ax, str):
                try:
                    axis_idx = mesh.axis_names.index(ax)
                    unreduced_refs.append(f'"{_MESH_IDX_PREFIX}{axis_idx}"')
                except ValueError:
                    pass
            elif isinstance(ax, int):
                unreduced_refs.append(f'"{_MESH_IDX_PREFIX}{ax}"')
        if unreduced_refs:
            unreduced_str = f", unreduced={{{', '.join(unreduced_refs)}}}"

    return f"#sdy.sharding_per_value<[<@mesh, [{dims_str}]{unreduced_str}>]>"


def _sharding_constraint(tensor, mesh, partition_spec, unreduced=None):
    sdy_sharding = _partition_spec_to_sdy_sharding(mesh, partition_spec, unreduced)
    return torch.ops.tt.sharding_constraint(tensor, sdy_sharding)


# ── Model building blocks ─────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = RMS_EPS):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.w * normed).to(x.dtype)


class LoRALinear(nn.Module):
    """Frozen base weight + trainable LoRA A/B matrices."""

    def __init__(self, in_f: int, out_f: int, rank: int, alpha: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f), requires_grad=False)
        self.bias_p = nn.Parameter(torch.empty(out_f), requires_grad=False) if bias else None
        self.lora_A = nn.Parameter(torch.empty(in_f, rank))  # trainable
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))  # trainable
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias_p)
        lora = (x @ self.lora_A @ self.lora_B) * self.scale
        return base + lora


def _frozen_linear(in_f: int, out_f: int, bias: bool = True) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=bias)
    for p in lin.parameters():
        p.requires_grad_(False)
    return lin


class Attention(nn.Module):
    def __init__(self, use_lora: bool):
        super().__init__()
        if use_lora:
            self.q_proj = LoRALinear(HIDDEN, Q_DIM, LORA_RANK, LORA_ALPHA_DEFAULT, bias=True)
            self.v_proj = LoRALinear(HIDDEN, KV_DIM, LORA_RANK, LORA_ALPHA_DEFAULT, bias=True)
        else:
            self.q_proj = _frozen_linear(HIDDEN, Q_DIM, bias=True)
            self.v_proj = _frozen_linear(HIDDEN, KV_DIM, bias=True)
        self.k_proj = _frozen_linear(HIDDEN, KV_DIM, bias=True)
        self.o_proj = _frozen_linear(Q_DIM, HIDDEN, bias=True)
        self.sinks = nn.Parameter(torch.zeros(N_HEADS), requires_grad=False)

    @staticmethod
    def _repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
        if n == 1:
            return x
        B, Hkv, S, D = x.shape
        return x[:, :, None].expand(B, Hkv, n, S, D).reshape(B, Hkv * n, S, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(x).view(B, S, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(x).view(B, S, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        k = self._repeat_kv(k, N_KV_GROUPS)
        v = self._repeat_kv(v, N_KV_GROUPS)

        scale = HEAD_DIM**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S, S]
        causal = torch.ones(S, S, device=x.device, dtype=torch.bool).triu(diagonal=1)
        attn = attn.masked_fill(causal[None, None], float("-inf"))
        attn = F.softmax(attn.float(), dim=-1).to(x.dtype)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, Q_DIM)
        return self.o_proj(out)


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(N_EXPERTS, HIDDEN), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(N_EXPERTS), requires_grad=False)

    def forward(self, hidden: torch.Tensor):
        flat = hidden.reshape(-1, HIDDEN)  # [T, H]
        logits = F.linear(flat, self.weight, self.bias)  # [T, E]
        top_v, top_idx = torch.topk(logits, N_TOP_K, dim=-1)  # [T, K]
        scores = F.softmax(top_v, dim=-1, dtype=top_v.dtype)  # [T, K]
        weights = torch.zeros_like(logits).scatter(1, top_idx, scores)  # [T, E]
        return weights


class MoEExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up = nn.Parameter(
            torch.empty(N_EXPERTS, HIDDEN, 2 * INTER, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.gate_up_bias = nn.Parameter(
            torch.zeros(N_EXPERTS, 2 * INTER, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.down = nn.Parameter(
            torch.empty(N_EXPERTS, INTER, HIDDEN, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.down_bias = nn.Parameter(
            torch.zeros(N_EXPERTS, HIDDEN, dtype=torch.bfloat16),
            requires_grad=False,
        )


class DecoderLayer(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        use_lora = layer_idx >= LORA_START_LAY
        self.input_norm = RMSNorm(HIDDEN)
        self.attn = Attention(use_lora)
        self.post_attn_norm = RMSNorm(HIDDEN)
        self.router = Router()
        self.experts = MoEExperts()
        self.layer_idx = layer_idx
        self.use_lora = use_lora


# ── MoE forward kernel ────────────────────────────────────────────────


def _moe_block(
    hidden: torch.Tensor,  # [B, S, H]
    routing_w: torch.Tensor,  # [T, E]  T = B*S
    gate_up: torch.Tensor,  # [E, H, 2I]
    gate_up_b: torch.Tensor,  # [E, 2I]
    down: torch.Tensor,  # [E, I, H]
    down_b: torch.Tensor,  # [E, H]
    mesh: Mesh,
) -> torch.Tensor:
    B, S, H = hidden.shape

    # Pin each weight to its expert shard on this device.
    gate_up = _sharding_constraint(gate_up, mesh, ("model", None, None))
    gate_up_b = _sharding_constraint(gate_up_b, mesh, ("model", None))
    down = _sharding_constraint(down, mesh, ("model", None, None))
    down_b = _sharding_constraint(down_b, mesh, ("model", None))

    # Gather expert shards via replicated layout.
    # Using (None, ...) keeps global E=32 fixed; explicit all_gather(dim=0)
    # would concatenate shards and inflate E to 128 on a 4-way mesh.
    gate_up = _sharding_constraint(gate_up, mesh, (None, None, None))
    gate_up_b = _sharding_constraint(gate_up_b, mesh, (None, None))
    down = _sharding_constraint(down, mesh, (None, None, None))
    down_b = _sharding_constraint(down_b, mesh, (None, None))

    flat = hidden.reshape(-1, H)  # [T, H]

    gu = torch.einsum("th,ehi->eti", flat, gate_up)  # [E, T, 2I]
    gu = gu + gate_up_b[:, None, :]

    gate, up = gu.chunk(2, dim=-1)  # [E, T, I] each (contiguous after deinterleave)
    gate = gate.clamp(max=MOE_LIMIT)
    up = up.clamp(min=-MOE_LIMIT, max=MOE_LIMIT)
    glu = gate * torch.sigmoid(gate * MOE_ALPHA)

    expert_out = torch.einsum("eti,eih->eth", (up + 1) * glu, down)  # [E, T, H]
    expert_out = expert_out + down_b[:, None, :]

    # routing_w: [T, E] → partial sum; Shardy inserts all-reduce
    out = torch.einsum("et,eth->th", routing_w.t(), expert_out)  # [T, H]

    return out.reshape(B, S, H)


# ── Main model ────────────────────────────────────────────────────────


class GptOss20B(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN)
        for p in self.embed_tokens.parameters():
            p.requires_grad_(False)
        self.layers = nn.ModuleList([DecoderLayer(i) for i in range(N_LAYERS)])
        self.norm = RMSNorm(HIDDEN)
        self.lm_head = _frozen_linear(HIDDEN, VOCAB_SIZE, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed_tokens.weight.data, std=0.02)
        for layer in self.layers:
            for p in [
                layer.experts.gate_up,
                layer.experts.down,
                layer.router.weight,
                layer.attn.k_proj.weight,
                layer.attn.o_proj.weight,
            ]:
                nn.init.kaiming_uniform_(p.data, a=math.sqrt(5))
            for proj in [layer.attn.q_proj, layer.attn.v_proj]:
                nn.init.kaiming_uniform_(proj.weight.data, a=math.sqrt(5))
                if hasattr(proj, "bias_p") and proj.bias_p is not None:
                    nn.init.zeros_(proj.bias_p.data)
                elif hasattr(proj, "bias") and proj.bias is not None:
                    nn.init.zeros_(proj.bias.data)
        nn.init.normal_(self.lm_head.weight.data, std=0.02)

    def deinterleave(self):
        """Convert gate_up layout from interleaved [g0,u0,...] to contiguous [gate...,up...].

        Run once on CPU before moving to device so there are no strided
        slices during the backward pass.
        """
        with torch.no_grad():
            for layer in self.layers:
                gu = layer.experts.gate_up.data  # [E, H, 2I]
                layer.experts.gate_up.data = torch.cat([gu[..., ::2].contiguous(), gu[..., 1::2].contiguous()], dim=-1)
                gb = layer.experts.gate_up_bias.data  # [E, 2I]
                layer.experts.gate_up_bias.data = torch.cat(
                    [gb[..., ::2].contiguous(), gb[..., 1::2].contiguous()], dim=-1
                )

    def shard(self, mesh: Mesh):
        """Apply Megatron-style TP sharding to all layers in-place."""
        for layer in self.layers:
            # Expert TP: shard on expert (first) dim
            xs.mark_sharding(layer.experts.gate_up, mesh, ("model", None, None))
            xs.mark_sharding(layer.experts.gate_up_bias, mesh, ("model", None))
            xs.mark_sharding(layer.experts.down, mesh, ("model", None, None))
            xs.mark_sharding(layer.experts.down_bias, mesh, ("model", None))

            # Router: replicated — prevents Shardy propagating E-sharding
            # onto routing_w, which would trigger all-gathers on scatter/gather ops
            xs.mark_sharding(layer.router.weight, mesh, (None, None))
            xs.mark_sharding(layer.router.bias, mesh, (None,))

            # Attention Megatron-style TP
            # Column-parallel: Q, K, V shard the output (heads) dim
            for proj in [layer.attn.q_proj, layer.attn.v_proj]:
                xs.mark_sharding(proj.weight, mesh, ("model", None))
                if getattr(proj, "bias_p", None) is not None:
                    xs.mark_sharding(proj.bias_p, mesh, ("model",))
                elif getattr(proj, "bias", None) is not None:
                    xs.mark_sharding(proj.bias, mesh, ("model",))
                # LoRA: A replicated [HIDDEN, rank]; B column-parallel [rank, out]
                if hasattr(proj, "lora_A"):
                    xs.mark_sharding(proj.lora_A, mesh, (None, None))
                    xs.mark_sharding(proj.lora_B, mesh, (None, "model"))

            xs.mark_sharding(layer.attn.k_proj.weight, mesh, ("model", None))
            xs.mark_sharding(layer.attn.k_proj.bias, mesh, ("model",))

            # Row-parallel: O shard input (Q_DIM) dim; Shardy inserts all-reduce
            xs.mark_sharding(layer.attn.o_proj.weight, mesh, (None, "model"))
            # o_proj bias [HIDDEN] is NOT sharded — added once after the all-reduce

            # Sinks [N_HEADS]: companion to column-parallel Q heads
            xs.mark_sharding(layer.attn.sinks, mesh, ("model",))

    def forward(self, input_ids: torch.Tensor, mesh: Mesh):
        """Forward pass without gradient tracking, saving per-layer inputs for recomputation.

        Args:
            input_ids: [B, S] integer token ids (on device)
            mesh: SPMD mesh

        Returns:
            (out, saved) where out is [B, S, H] after final norm and
            saved is a list of per-layer input hidden states.
        """
        saved = []
        with torch.no_grad():
            hidden = self.embed_tokens(input_ids).to(torch.bfloat16)

            for layer in self.layers:
                saved.append(hidden)

                h = layer.input_norm(hidden)
                h = layer.attn(h)
                hidden = hidden + h

                h_norm = layer.post_attn_norm(hidden)
                rw = layer.router(h_norm)
                moe = _moe_block(
                    h_norm,
                    rw,
                    layer.experts.gate_up,
                    layer.experts.gate_up_bias,
                    layer.experts.down,
                    layer.experts.down_bias,
                    mesh,
                )
                hidden = hidden + moe

                xm.mark_step()

            hidden = self.norm(hidden)
            torch_xla.sync(wait=True)

        return hidden, saved

    def backward(self, saved: list, grad_out: torch.Tensor, mesh: Mesh):
        """Per-layer recompute backward accumulating LoRA gradients.

        Recomputes each layer's forward under grad, runs autograd backward,
        then syncs to free intermediates before moving to the next layer.
        """
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            h = saved[i].detach().requires_grad_(True)

            with torch.enable_grad():
                h_normed = layer.input_norm(h)
                attn_out = layer.attn(h_normed)
                h_post_a = h + attn_out

                h_normed2 = layer.post_attn_norm(h_post_a)
                rw = layer.router(h_normed2)

                gu = layer.experts.gate_up.detach()
                gb = layer.experts.gate_up_bias.detach()
                dn = layer.experts.down.detach()
                db = layer.experts.down_bias.detach()
                xs.mark_sharding(gu, mesh, ("model", None, None))
                xs.mark_sharding(dn, mesh, ("model", None, None))

                moe_out = _moe_block(h_normed2, rw, gu, gb, dn, db, mesh)
                h_out = h_post_a + moe_out

            torch.autograd.backward(h_out, grad_out)
            torch_xla.sync(wait=True)

            # h_out includes residuals so h.grad IS d(loss)/d(h_in) directly
            if h.grad is not None:
                grad_out = h.grad.detach()

            torch_xla.sync(wait=True)

    def lora_params(self):
        """Yield (name, layer_idx, param) for all trainable LoRA parameters."""
        for layer in self.layers:
            if layer.use_lora:
                for name, p in layer.attn.named_parameters():
                    if "lora_" in name:
                        yield name, layer.layer_idx, p


# ── SPMD / mesh setup ─────────────────────────────────────────────────


def setup_spmd():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def create_mesh() -> Mesh:
    n = xr.global_runtime_device_count()
    shapes = {4: (1, 4), 8: (2, 4), 32: (8, 4)}
    if n not in shapes:
        raise RuntimeError(f"Unsupported device count: {n}. Expected 4, 8, or 32.")
    return Mesh(np.arange(n), shapes[n], ("batch", "model"))
