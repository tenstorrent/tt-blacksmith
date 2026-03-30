# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re
from collections import OrderedDict
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

    self._dbg = OrderedDict()

    flat = _keep(self._dbg, "router_input", hidden_states.reshape(-1, self.hidden_dim)).float()

    router_logits = _keep(self._dbg, "router_logits",
                          F.linear(flat, self.weight.float(), self.bias.float()))

    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
    self._dbg["topk_indices_raw"] = router_indices
    #router_indices = router_indices.clamp(0, 31)
    router_top_value = _keep(self._dbg, "topk_values_pre_softmax", router_top_value)

    router_top_value = _keep(self._dbg, "topk_values_post_softmax",
                             torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype))
    self._dbg["topk_post_hook_grad"] = [None]
    if router_top_value.requires_grad:
        _captured = self._dbg["topk_post_hook_grad"]
        router_top_value.register_hook(lambda g: _captured.__setitem__(0, g.detach().cpu()))

    router_scores = _keep(self._dbg, "routing_weights",
                          torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value))

    output_router_scores = _keep(self._dbg, "output_router_scores", router_scores)

    return output_router_scores.to(hidden_states.dtype), router_indices


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


def verify_scatter_backward(router, layer_idx):
    if not hasattr(router, "_dbg"):
        return
    
    routing_weights = router._dbg.get("routing_weights")
    topk_post = router._dbg.get("topk_values_post_softmax")
    indices = router._dbg.get("topk_indices_raw")
    if layer_idx == 15:
        # save the inputs to the scatter for later analysis
        print("Saving scatter debug tensors for layer 15...")
        save_dict = {
            "routing_weights": routing_weights.cpu(),
            "topk_post": topk_post.cpu(),
            "indices": indices.cpu(),
        }
        if routing_weights.grad is not None:
            save_dict["gradient_input"] = router._dbg["output_router_scores"].grad.cpu()
        # if file exists, append a number suffix to avoid overwriting
        import os
        filename = f"scatter_debug_layer{layer_idx}.pt"
        if os.path.exists(filename):
            i = 1
            while os.path.exists(f"scatter_debug_layer{layer_idx}_{i}.pt"):
                i += 1
            filename = f"scatter_debug_layer{layer_idx}_{i}.pt"
        torch.save(save_dict, filename)
    else:
        print(f"Layer {layer_idx} scatter debug tensors not saved (only layer 15 is saved)")
    
    topk_pre = router._dbg.get("topk_values_pre_softmax")
    if any(x is None for x in [routing_weights, topk_post, topk_pre, indices]):
        print("  scatter verify: missing tensors")
        return
    hook_captured = router._dbg.get("topk_post_hook_grad", [None])[0]
    if routing_weights.grad is None:
        print("  scatter verify: routing_weights.grad not available")
        return

    # Step 1: scatter backward = gather
    grad_upstream = routing_weights.grad.float().cpu()
    indices_cpu = indices.cpu()
    expected_scatter_bwd = torch.gather(grad_upstream, 1, indices_cpu)

    print(f"  scatter verify — scatter_bwd (cpu gather): norm={expected_scatter_bwd.norm():.6e}, max={expected_scatter_bwd.abs().max():.6e}")

    # Step 2: check topk_post.grad (retain_grad) and hook against scatter_bwd
    if topk_post.grad is not None:
        actual_post = topk_post.grad.float().cpu()
        print(f"  scatter verify — topk_post.grad (retain): norm={actual_post.norm():.6e}, max={actual_post.abs().max():.6e}, match={torch.allclose(expected_scatter_bwd, actual_post, atol=1e-3, rtol=1e-3)}")
    if hook_captured is not None:
        hg = hook_captured.float()
        print(f"  scatter verify — topk_post.grad (hook):   norm={hg.norm():.6e}, max={hg.abs().max():.6e}, match={torch.allclose(expected_scatter_bwd, hg, atol=1e-3, rtol=1e-3)}")
    else:
        print(f"  scatter verify — topk_post.grad (hook):   not captured")

    # Step 3: propagate expected gradient through softmax backward and compare with topk_pre.grad
    # softmax_bwd: dx = y * (dy - sum(y * dy, dim=-1, keepdim=True))
    if topk_pre.grad is not None:
        y = topk_post.float().cpu()
        dy = expected_scatter_bwd
        expected_pre_grad = y * (dy - (y * dy).sum(dim=-1, keepdim=True))
        actual_pre_grad = topk_pre.grad.float().cpu()
        print(f"  scatter+softmax verify — expected topk_pre.grad: norm={expected_pre_grad.norm():.6e}, max={expected_pre_grad.abs().max():.6e}")
        print(f"  scatter+softmax verify — actual  topk_pre.grad:  norm={actual_pre_grad.norm():.6e}, max={actual_pre_grad.abs().max():.6e}")
        print(f"  scatter+softmax verify — match: {torch.allclose(expected_pre_grad, actual_pre_grad, atol=1e-3, rtol=1e-3)}, max abs diff: {(expected_pre_grad - actual_pre_grad).abs().max():.6e}")
    else:
        print(f"  scatter+softmax verify — topk_pre.grad not available")



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
                print(f"  experts.{name}: {_stats(t.grad)}, {_stats(t)}", flush=True)
            else:
                print(f"  experts.{name}: grad=None", flush=True)

    if hasattr(router, "_dbg"):
        for name in ("routing_weights", "topk_values_post_softmax",
                      "topk_values_pre_softmax", "router_logits", "router_input"):
            t = router._dbg.get(name)
            if t is not None and t.grad is not None:
                print(f"  router.{name}: grad_stats> {_stats(t.grad)}; fwd_stats>{_stats(t)}", flush=True)
            else:
                print(f"  router.{name}: grad=None", flush=True)

        raw_idx = router._dbg.get("topk_indices_raw")
        if raw_idx is not None:
            # print them all
            print(f"  router.topk_indices_raw: shape={raw_idx.shape}, values={raw_idx.cpu().numpy()}", flush=True)

    verify_scatter_backward(router, layer_idx)
    print(f"{'='*80}\n", flush=True)
