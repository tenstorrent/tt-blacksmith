# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Isolated single-chip MoE repro for layer 16 of GPT-OSS 20B.

Router + deinterleaved-BMM experts block, run directly on XLA tensors
(same as repro_big.py pattern). Weights copied from the real model.
Random hidden_states in, compare CPU vs TT grads via PCC.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.quantization_config import Mxfp4Config

from blacksmith.models.torch.gpt_oss_overrides import _deinterleave_expert_weights

xr.set_device_type("TT")

TARGET_LAYER = 16
MODEL_NAME = "openai/gpt-oss-20b"
DTYPE = torch.bfloat16
BATCH_SIZE = 1
SEQ_LEN = 64
SEED = 42


class MoEBlock(nn.Module):
    """Router + deinterleaved experts. Exact same ops as GptOssMLP + override."""

    def __init__(self, num_experts, hidden_size, expert_dim, top_k, alpha, limit):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.expert_dim = expert_dim
        self.alpha = alpha
        self.limit = limit

        self.router_weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.router_bias = nn.Parameter(torch.empty(num_experts))
        self.gate_proj = nn.Parameter(torch.empty(num_experts, hidden_size, expert_dim))
        self.up_proj = nn.Parameter(torch.empty(num_experts, hidden_size, expert_dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, expert_dim, hidden_size))
        self.gate_proj_bias = nn.Parameter(torch.empty(num_experts, expert_dim))
        self.up_proj_bias = nn.Parameter(torch.empty(num_experts, expert_dim))
        self.down_proj_bias = nn.Parameter(torch.empty(num_experts, hidden_size))

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        self._intermediates = {}

        def keep(name, t):
            t.retain_grad()
            self._intermediates[name] = t
            return t

        flat = keep("flat", hidden_states.reshape(-1, self.hidden_size))
        router_logits = keep("router_logits", F.linear(flat, self.router_weight, self.router_bias))
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = keep("router_top_value_pre_softmax", router_top_value)
        router_top_value = keep("router_top_value_post_softmax",
                                F.softmax(router_top_value, dim=1, dtype=router_top_value.dtype))
        routing_weights = keep("routing_weights",
                               torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value))

        num_experts = routing_weights.shape[1]
        x = keep("x_repeated", flat.repeat(num_experts, 1).view(num_experts, -1, self.hidden_size))

        gate_raw = keep("gate_raw", torch.bmm(x, self.gate_proj) + self.gate_proj_bias[..., None, :])
        up_raw = keep("up_raw", torch.bmm(x, self.up_proj) + self.up_proj_bias[..., None, :])

        gate = keep("gate_clamped", gate_raw.clamp(min=None, max=self.limit))
        up = keep("up_clamped", up_raw.clamp(min=-self.limit, max=self.limit))

        sigmoid_gate = keep("sigmoid_gate", torch.sigmoid(gate * self.alpha))
        glu = keep("glu", gate * sigmoid_gate)
        inner = keep("inner", (up + 1) * glu)

        expert_out = keep("expert_out_pre_bias",
                          torch.bmm(inner, self.down_proj))
        expert_out = keep("expert_out",
                          expert_out + self.down_proj_bias[..., None, :])

        expert_out = keep("expert_out_viewed",
                          expert_out.view(num_experts, batch_size, -1, self.hidden_size))

        routing_view = keep("routing_view",
                            routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None])

        weighted = keep("weighted", expert_out * routing_view)
        output = keep("output", weighted.sum(dim=0))

        return output


def load_moe_weights_from_model(moe_block, model, layer_idx):
    """Copy layer's router + deinterleaved expert weights into our MoEBlock."""
    layer = model.model.layers[layer_idx]
    router = layer.mlp.router
    experts = layer.mlp.experts

    with torch.no_grad():
        moe_block.router_weight.copy_(router.weight.data)
        moe_block.router_bias.copy_(router.bias.data)
        moe_block.gate_proj.copy_(experts.gate_proj.data)
        moe_block.up_proj.copy_(experts.up_proj.data)
        moe_block.down_proj.copy_(experts.down_proj.data)
        moe_block.gate_proj_bias.copy_(experts.gate_proj_bias.data)
        moe_block.up_proj_bias.copy_(experts.up_proj_bias.data)
        moe_block.down_proj_bias.copy_(experts.down_proj_bias.data)


def forward_backward(moe, x, device):
    """Forward + backward, return output and all grads (repro_big.py pattern)."""
    moe.to(device)
    moe.train()

    x_dev = x.clone().to(device)
    x_dev.requires_grad_(True)
    x_dev.retain_grad()

    out = moe(x_dev)
    loss = out.sum()
    loss.backward()

    if device != torch.device("cpu"):
        torch_xla.sync(wait=True)

    grads = {}
    for name, p in moe.named_parameters():
        grads[name] = p.grad.detach().cpu() if p.grad is not None else None

    intermediate_grads = {}
    intermediate_grads["input"] = x_dev.grad.detach().cpu() if x_dev.grad is not None else None
    for name, t in moe._intermediates.items():
        intermediate_grads[name] = t.grad.detach().cpu() if t.grad is not None else None

    return out.detach().cpu(), grads, intermediate_grads


def forward_backward_raw(moe, x, device):
    """Same computation but with raw tensors on device (repro_big.py style, no nn.Module.forward)."""
    params = {}
    for name, p in moe.named_parameters():
        params[name] = p.data.clone().to(device).requires_grad_(True)

    x_dev = x.clone().to(device)
    x_dev.requires_grad_(True)

    intermediates = {}

    def keep(name, t):
        t.retain_grad()
        intermediates[name] = t
        return t

    flat = keep("flat", x_dev.reshape(-1, moe.hidden_size))
    router_logits = keep("router_logits",
                         F.linear(flat, params["router_weight"], params["router_bias"]))
    router_top_value, router_indices = torch.topk(router_logits, moe.top_k, dim=-1)
    router_top_value = keep("router_top_value_pre_softmax", router_top_value)
    router_top_value = keep("router_top_value_post_softmax",
                            F.softmax(router_top_value, dim=1, dtype=router_top_value.dtype))
    routing_weights = keep("routing_weights",
                           torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value))

    num_experts = routing_weights.shape[1]
    batch_size = x_dev.shape[0]
    x_rep = keep("x_repeated", flat.repeat(num_experts, 1).view(num_experts, -1, moe.hidden_size))

    gate_raw = keep("gate_raw",
                    torch.bmm(x_rep, params["gate_proj"]) + params["gate_proj_bias"][..., None, :])
    up_raw = keep("up_raw",
                  torch.bmm(x_rep, params["up_proj"]) + params["up_proj_bias"][..., None, :])

    gate = keep("gate_clamped", gate_raw.clamp(min=None, max=moe.limit))
    up = keep("up_clamped", up_raw.clamp(min=-moe.limit, max=moe.limit))

    sigmoid_gate = keep("sigmoid_gate", torch.sigmoid(gate * moe.alpha))
    glu = keep("glu", gate * sigmoid_gate)
    inner = keep("inner", (up + 1) * glu)

    expert_out = keep("expert_out_pre_bias", torch.bmm(inner, params["down_proj"]))
    expert_out = keep("expert_out", expert_out + params["down_proj_bias"][..., None, :])
    expert_out = keep("expert_out_viewed",
                      expert_out.view(num_experts, batch_size, -1, moe.hidden_size))

    routing_view = keep("routing_view",
                        routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None])

    weighted = keep("weighted", expert_out * routing_view)
    output = keep("output", weighted.sum(dim=0))

    loss = output.sum()
    loss.backward()

    torch_xla.sync(wait=True)

    grads = {}
    for name, p in params.items():
        grads[name] = p.grad.detach().cpu() if p.grad is not None else None

    intermediate_grads = {}
    intermediate_grads["input"] = x_dev.grad.detach().cpu() if x_dev.grad is not None else None
    for name, t in intermediates.items():
        intermediate_grads[name] = t.grad.detach().cpu() if t.grad is not None else None

    return output.detach().cpu(), grads, intermediate_grads


def compute_pcc(a, b):
    a_f, b_f = a.flatten().float(), b.flatten().float()
    nz = (a_f.abs() > 1e-12) | (b_f.abs() > 1e-12)
    if nz.sum() < 2:
        return float("nan")
    return torch.corrcoef(torch.stack([a_f[nz], b_f[nz]]))[0, 1].item()


def stats(t):
    f = t.float()
    return (
        f"min={f.min().item():.6e} max={f.max().item():.6e} "
        f"norm={f.norm().item():.6e} "
        f"inf={torch.isinf(f).sum().item()} nan={torch.isnan(f).sum().item()}"
    )


def main():
    torch.manual_seed(SEED)

    # ---- Load full model, deinterleave, extract layer 16 weights ----
    print(f"Loading {MODEL_NAME}...", flush=True)
    quantization_config = Mxfp4Config(dequantize=True)
    model_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=model_config,
        quantization_config=quantization_config,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_mod
    target_experts = model.model.layers[TARGET_LAYER].mlp.experts
    assert isinstance(target_experts, gpt_oss_mod.GptOssExperts)
    _deinterleave_expert_weights(target_experts)

    router = model.model.layers[TARGET_LAYER].mlp.router
    num_experts = router.num_experts
    hidden_size = router.hidden_dim
    expert_dim = target_experts.intermediate_size
    top_k = router.top_k
    alpha = target_experts.alpha
    limit = target_experts.limit

    print(f"Layer {TARGET_LAYER}: num_experts={num_experts}, hidden_size={hidden_size}, "
          f"expert_dim={expert_dim}, top_k={top_k}, alpha={alpha}, limit={limit}", flush=True)

    # Build two identical MoE blocks
    import copy
    moe_cpu = MoEBlock(num_experts, hidden_size, expert_dim, top_k, alpha, limit)
    load_moe_weights_from_model(moe_cpu, model, TARGET_LAYER)
    moe_cpu.to(DTYPE)

    moe_tt = copy.deepcopy(moe_cpu)

    del model
    print("Model freed.\n", flush=True)

    # ---- Random input ----
    torch.manual_seed(SEED)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, hidden_size, dtype=DTYPE)

    # ---- CPU forward+backward ----
    print(f"{'='*80}")
    print("CPU FORWARD+BACKWARD")
    print(f"{'='*80}")
    out_cpu, grads_cpu, inter_cpu = forward_backward(moe_cpu, x, torch.device("cpu"))
    print(f"  output: {stats(out_cpu)}", flush=True)
    print("  -- intermediate grads (backward order, last computed first) --")
    for name, g in inter_cpu.items():
        print(f"  d/{name}: {stats(g) if g is not None else 'None'}", flush=True)
    print("  -- parameter grads --")
    for name, g in grads_cpu.items():
        print(f"  d/{name}: {stats(g) if g is not None else 'None'}", flush=True)

    # ---- TT forward+backward ----
    print(f"\n{'='*80}")
    print("TT FORWARD+BACKWARD")
    print(f"{'='*80}")
    device = xm.xla_device()

    out_tt, grads_tt, inter_tt = forward_backward_raw(moe_tt, x, device)
    print(f"  output: {stats(out_tt)}", flush=True)
    print("  -- intermediate grads --")
    for name, g in inter_tt.items():
        print(f"  d/{name}: {stats(g) if g is not None else 'None'}", flush=True)
    print("  -- parameter grads --")
    for name, g in grads_tt.items():
        print(f"  d/{name}: {stats(g) if g is not None else 'None'}", flush=True)

    # ---- PCC ----
    print(f"\n{'='*80}")
    print("PCC COMPARISON  (intermediates)")
    print(f"{'='*80}")
    for name in inter_cpu:
        g_c, g_t = inter_cpu[name], inter_tt.get(name)
        if g_c is not None and g_t is not None:
            pcc = compute_pcc(g_c, g_t)
            print(f"  {name:35s}  PCC={pcc:.8f}  "
                  f"inf_cpu={torch.isinf(g_c.float()).sum().item()}  "
                  f"inf_tt={torch.isinf(g_t.float()).sum().item()}", flush=True)
        else:
            print(f"  {name:35s}  MISSING", flush=True)

    print(f"\n{'='*80}")
    print("PCC COMPARISON  (parameters)")
    print(f"{'='*80}")
    for name in grads_cpu:
        g_c, g_t = grads_cpu[name], grads_tt[name]
        if g_c is not None and g_t is not None:
            pcc = compute_pcc(g_c, g_t)
            print(f"  {name:35s}  PCC={pcc:.8f}  "
                  f"inf_cpu={torch.isinf(g_c.float()).sum().item()}  "
                  f"inf_tt={torch.isinf(g_t.float()).sum().item()}", flush=True)
        else:
            print(f"  {name:35s}  MISSING", flush=True)

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
