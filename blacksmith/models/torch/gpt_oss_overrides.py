# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re

import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.quantization_config import Mxfp4Config


def get_gpt_oss_model(config, device, debug_router_grads=False):
    """Load GPT-OSS model with deinterleaving overrides, LoRA, and compilation."""
    quantization_config = Mxfp4Config(dequantize=True)

    model_config = AutoConfig.from_pretrained(
        config.model_name, trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        config=model_config,
        quantization_config=quantization_config,
        torch_dtype=eval(config.dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    override_gpt_oss_experts_deinterleave(model, debug_router_grads=debug_router_grads)

    if config.training_type == "lora":
        n = model.config.num_hidden_layers
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            layers_to_transform=list(range(n//2, n)),
            task_type=config.lora_task_type,
        )
        model = get_peft_model(model, lora_config)

    torch._dynamo.config.recompile_limit = 100

    print("\n=== MODEL NAMED MODULES (with .weight) ===")
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            print(f"  {name:80s} weight {tuple(module.weight.shape)}")
    print("=" * 60 + "\n")

    model.to(device)

    if config.use_tt:
        compile_options = {
            "tt_enable_torch_fx_fusion_pass": False,
            "tt_legacy_compile": True,
        }
        model = torch.compile(model, backend="tt", options=compile_options)

    return model


def shard_params_by_pattern(model, mesh, patterns):
    """Shard parameters by regex matching on named_parameters()."""
    if mesh is None or not patterns:
        return
    for name, param in model.named_parameters():
        for pattern, spec in patterns:
            if re.search(pattern, name):
                xs.mark_sharding(param, mesh, tuple(spec))
                break
    torch_xla.sync(wait=True)


# ---------------------------------------------------------------------------
# Expert weight deinterleaving + batched BMM forward override
# ---------------------------------------------------------------------------


def override_gpt_oss_experts_deinterleave(model, debug_router_grads=False):
    """
    De-interleave gate_up_proj into separate gate_proj and up_proj, and
    patch forward to use the batched BMM path for both training and inference.
    """
    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss_mod

    for module in model.modules():
        if isinstance(module, gpt_oss_mod.GptOssExperts):
            _deinterleave_expert_weights(module)
        if debug_router_grads and isinstance(module, gpt_oss_mod.GptOssTopKRouter):
            module.forward = _debug_router_forward.__get__(module, type(module))


def _deinterleave_expert_weights(experts):
    with torch.no_grad():
        gate_proj_data = experts.gate_up_proj.data[:, :, ::2].contiguous()
        up_proj_data = experts.gate_up_proj.data[:, :, 1::2].contiguous()
        gate_bias_data = experts.gate_up_proj_bias.data[:, ::2].contiguous()
        up_bias_data = experts.gate_up_proj_bias.data[:, 1::2].contiguous()

    del experts.gate_up_proj
    del experts.gate_up_proj_bias

    experts.gate_proj = nn.Parameter(gate_proj_data)
    experts.up_proj = nn.Parameter(up_proj_data)
    experts.gate_proj_bias = nn.Parameter(gate_bias_data)
    experts.up_proj_bias = nn.Parameter(up_bias_data)

    experts.forward = _deinterleaved_experts_forward.__get__(experts, type(experts))


def _keep(dbg, name, t):
    if t.requires_grad:
        t.retain_grad()
    dbg[name] = t
    return t


def _debug_router_forward(self, hidden_states):
    """Router forward with retain_grad on every intermediate."""
    import torch.nn.functional as F

    self._dbg = {}

    flat = _keep(self._dbg, "router_input", hidden_states.reshape(-1, self.hidden_dim))

    router_logits = _keep(self._dbg, "router_logits",
                          F.linear(flat, self.weight, self.bias))

    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
    #self._dbg["topk_indices_raw"] = router_indices
    #router_indices = router_indices.clamp(0, 31)
    router_top_value = _keep(self._dbg, "topk_values_pre_softmax", router_top_value)

    router_top_value = _keep(self._dbg, "topk_values_post_softmax",
                             torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype))

    router_scores = _keep(self._dbg, "routing_weights",
                          torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value))

    return router_scores, router_indices


def _deinterleaved_experts_forward(
    self, hidden_states, router_indices=None, routing_weights=None
):
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    num_experts = routing_weights.shape[1]

    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)

    gate = torch.bmm(hidden_states, self.gate_proj) + self.gate_proj_bias[..., None, :]
    up = torch.bmm(hidden_states, self.up_proj) + self.up_proj_bias[..., None, :]

    gate = gate.clamp(min=None, max=self.limit)
    up = up.clamp(min=-self.limit, max=self.limit)
    glu = gate * torch.sigmoid(gate * self.alpha)
    next_states = torch.bmm(((up + 1) * glu), self.down_proj)
    next_states = next_states + self.down_proj_bias[..., None, :]

    self._dbg = {}

    expert_out = _keep(self._dbg, "expert_out",
                       next_states.view(num_experts, batch_size, -1, self.hidden_size))

    routing_view = _keep(self._dbg, "routing_view",
                         routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None])

    weighted = _keep(self._dbg, "weighted", expert_out * routing_view)

    output = _keep(self._dbg, "output", weighted.sum(dim=0))

    return output


def print_debug_intermediates(model, layer_idx):
    """Print retained intermediate grads for the router backward branch."""
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    layer = base.model.layers[layer_idx]
    router = layer.mlp.router
    experts = layer.mlp.experts

    def _stats(t):
        f = t.float()
        return (
            f"shape={list(t.shape)}, "
            f"min={f.min().item():.6e}, max={f.max().item():.6e}, "
            f"norm={f.norm().item():.6e}, "
            f"inf={torch.isinf(f).sum().item()}, nan={torch.isnan(f).sum().item()}"
        )

    print(f"\n{'='*80}", flush=True)
    print(f"INTERMEDIATE GRADS — layer {layer_idx} router backward branch", flush=True)
    print(f"{'='*80}", flush=True)

    if hasattr(experts, "_dbg"):
        for name in ("output", "weighted", "routing_view", "expert_out"):
            t = experts._dbg.get(name)
            if t is not None and t.grad is not None:
                print(f"  experts.{name}: {_stats(t.grad)}", flush=True)
            else:
                print(f"  experts.{name}: grad=None", flush=True)

    if hasattr(router, "_dbg"):
        for name in ("routing_weights", "topk_values_post_softmax",
                      "topk_values_pre_softmax", "router_logits", "router_input"):
            t = router._dbg.get(name)
            if t is not None and t.grad is not None:
                print(f"  router.{name}: {_stats(t.grad)}", flush=True)
            else:
                print(f"  router.{name}: grad=None", flush=True)

        raw_idx = router._dbg.get("topk_indices_raw")
        if raw_idx is not None:
            lo = raw_idx.min().item()
            hi = raw_idx.max().item()
            oob = ((raw_idx < 0) | (raw_idx > 31)).sum().item()
            print(f"  router.topk_indices_raw: shape={list(raw_idx.shape)}, "
                  f"min={lo}, max={hi}, oob_count={oob}", flush=True)
            if oob > 0:
                torch.set_printoptions(threshold=10000, linewidth=200)
                print(f"  *** OUT-OF-RANGE INDICES DETECTED! ***", flush=True)
                bad_rows = (raw_idx < 0) | (raw_idx > 31)
                for r in range(raw_idx.shape[0]):
                    if bad_rows[r].any():
                        print(f"    row {r}: {raw_idx[r].tolist()}", flush=True)
                torch.set_printoptions(profile="default")

    print(f"{'='*80}\n", flush=True)
