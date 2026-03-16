# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import gc

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

from blacksmith.experiments.torch.gpt_oss.configs import TrainingConfig


class GatherTokens(torch.autograd.Function):
    """All-gather token activations with a symmetric backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.local_tokens = x.shape[0]
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)

        local_tokens_t = torch.tensor([x.shape[0]], dtype=torch.int64, device=x.device)
        gathered_tokens = [torch.zeros_like(local_tokens_t) for _ in range(ctx.world_size)]
        dist.all_gather(gathered_tokens, local_tokens_t, group=group)
        token_counts = [int(tokens.item()) for tokens in gathered_tokens]
        if len(set(token_counts)) != 1:
            raise RuntimeError(f"GatherTokens requires equal token counts across ranks, got {token_counts}")

        gathered_x = [torch.empty_like(x) for _ in range(ctx.world_size)]
        dist.all_gather(gathered_x, x.contiguous(), group=group)
        return torch.cat(gathered_x, dim=0)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        grad_all = grad_output.contiguous()
        dist.all_reduce(grad_all, op=dist.ReduceOp.SUM, group=ctx.group)
        start = ctx.rank * ctx.local_tokens
        grad_input = grad_all.narrow(0, start, ctx.local_tokens).contiguous()
        return grad_input, None


class AllReduceSum(torch.autograd.Function):
    """Differentiable all-reduce sum."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.group = group
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        grad_input = grad_output.contiguous()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_input, None


class ExpertParallelMLP(nn.Module):
    """Drop-in replacement for GptOssMLP with Expert Parallelism.

    Adapted from GptOssMLP + GptOssExperts in modeling_gpt_oss.py.

    Rank `r` owns experts `[r * n_local, (r + 1) * n_local)`. Every rank
    gathers the same token activations, applies the replicated router, runs
    only the token-expert pairs that map to its local experts, and
    all-reduces the accumulated partial outputs.
    """

    def __init__(
        self,
        original_mlp: GptOssMLP,
        ep_group: dist.ProcessGroup,
        module_name: str,
    ) -> None:
        super().__init__()
        self.ep_group = ep_group
        self.world_size = dist.get_world_size(ep_group)
        self.rank = dist.get_rank(ep_group)
        self.module_name = module_name

        # Router is replicated.
        self.router = original_mlp.router
        self.top_k = original_mlp.router.top_k

        orig_exp = original_mlp.experts
        self.num_experts_global = orig_exp.num_experts
        self.n_local = self.num_experts_global // self.world_size
        assert self.num_experts_global % self.world_size == 0, (
            f"num_experts ({self.num_experts_global}) must be divisible " f"by world_size ({self.world_size})"
        )

        low_expert = self.rank * self.n_local
        high_expert = low_expert + self.n_local

        fused_gate_up_proj = orig_exp.gate_up_proj.data[low_expert:high_expert]
        fused_gate_up_proj_bias = orig_exp.gate_up_proj_bias.data[low_expert:high_expert]

        # Deinterleave gate and up proj to make backward pass simpler.
        self.gate_proj = nn.Parameter(fused_gate_up_proj[..., ::2].clone().contiguous())
        self.up_proj = nn.Parameter(fused_gate_up_proj[..., 1::2].clone().contiguous())
        self.gate_proj_bias = nn.Parameter(fused_gate_up_proj_bias[..., ::2].clone().contiguous())
        self.up_proj_bias = nn.Parameter(fused_gate_up_proj_bias[..., 1::2].clone().contiguous())
        self.down_proj = nn.Parameter(orig_exp.down_proj.data[low_expert:high_expert].clone())
        self.down_proj_bias = nn.Parameter(orig_exp.down_proj_bias.data[low_expert:high_expert].clone())

        self.alpha = orig_exp.alpha
        self.limit = orig_exp.limit
        self.hidden_size = orig_exp.hidden_size
        self.intermediate_size = self.gate_proj.shape[-1]

    def __repr__(self) -> str:
        return (
            f"ExpertParallelMLP("
            f"rank={self.rank}, "
            f"experts={self.n_local}/{self.num_experts_global}, "
            f"hidden_size={self.hidden_size})"
        )

    def _run_local_experts(
        self,
        hidden_states: torch.Tensor,
        local_expert_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run token-expert assignments through this rank's local experts.

        Args:
            hidden_states: Tokens assigned to this rank, shape [M, H].
            local_expert_ids: Local expert index per token, shape [M],
                values in [0, n_local).
            weights: Per-token routing weight, shape [M].

        Returns:
            Weighted expert outputs, shape [M, H].
        """
        M, H = hidden_states.shape
        next_states = torch.zeros(M, H, dtype=hidden_states.dtype, device=hidden_states.device)
        if M == 0:
            return next_states

        with torch.no_grad():
            sort_idx = local_expert_ids.argsort(stable=True)
            sorted_local_ids = local_expert_ids.index_select(0, sort_idx)
            expert_counts = torch.bincount(sorted_local_ids, minlength=self.n_local)
            expert_offsets = torch.cumsum(expert_counts, dim=0) - expert_counts
            active_experts = torch.nonzero(expert_counts, as_tuple=False).squeeze(1)

        sorted_hidden_states = hidden_states.index_select(0, sort_idx)
        sorted_weights = weights.index_select(0, sort_idx)

        for local_e_tensor in active_experts:
            local_e = int(local_e_tensor.item())
            token_count = int(expert_counts[local_e].item())
            start = int(expert_offsets[local_e].item())
            token_idx = sort_idx.narrow(0, start, token_count)

            current_state = sorted_hidden_states.narrow(0, start, token_count)
            gate = current_state @ self.gate_proj.select(0, local_e) + self.gate_proj_bias.select(0, local_e)
            up = current_state @ self.up_proj.select(0, local_e) + self.up_proj_bias.select(0, local_e)
            gate = gate.clamp(max=self.limit)
            up = up.clamp(-self.limit, self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            out = ((up + 1) * glu) @ self.down_proj.select(0, local_e) + self.down_proj_bias.select(0, local_e)
            next_states.index_add_(
                0,
                token_idx,
                (out * sorted_weights.narrow(0, start, token_count).unsqueeze(1)).to(hidden_states.dtype),
            )

        return next_states

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)
        local_tokens = x.shape[0]

        # Gather tokens across all ranks.
        global_x = GatherTokens.apply(x, self.ep_group)
        total_tokens = global_x.shape[0]
        rank_slice_start = self.rank * local_tokens

        router_scores, router_indices = self.router.forward(global_x)
        local_router_scores = router_scores.narrow(0, rank_slice_start, local_tokens)
        partial_output = torch.zeros(total_tokens, H, dtype=global_x.dtype, device=global_x.device)

        # This rank's experts.
        low_expert = self.rank * self.n_local
        high_expert = low_expert + self.n_local

        for k in range(self.top_k):
            expert_ids = router_indices.select(1, k)
            local_mask = (expert_ids >= low_expert) & (expert_ids < high_expert)
            local_token_idx = torch.nonzero(local_mask, as_tuple=False).squeeze(1)

            if local_token_idx.numel() == 0:
                continue

            local_hidden = global_x.index_select(0, local_token_idx)
            local_ids = expert_ids.index_select(0, local_token_idx) - low_expert
            local_weights = router_scores.gather(1, expert_ids.unsqueeze(1)).squeeze(1)
            local_weights = local_weights.index_select(0, local_token_idx)

            local_out = self._run_local_experts(local_hidden, local_ids, local_weights)
            partial_output.index_add_(0, local_token_idx, local_out)

        reduced_output = AllReduceSum.apply(partial_output, self.ep_group)
        local_output = reduced_output.narrow(0, rank_slice_start, local_tokens).contiguous()
        return local_output.reshape(B, S, H), local_router_scores


def apply_expert_parallel(model: nn.Module, ep_group: dist.ProcessGroup) -> nn.Module:
    """Replace every GptOssMLP in the model with ExpertParallelMLP.

    Args:
        model: The loaded GptOss model.
        ep_group: Process group whose members share the expert bank.

    Returns:
        The model with all GptOssMLP layers replaced in-place.
    """
    for name, module in model.named_modules():
        if not isinstance(module, GptOssMLP):
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], ExpertParallelMLP(module, ep_group, module_name=name))
    return model


def build_ep_model(
    config: TrainingConfig,
    ep_group: dist.ProcessGroup,
    device: torch.device,
) -> nn.Module:
    """Load GPT OSS, apply Expert Parallelism and LoRA, move to device.

    Loads one rank at a time to avoid exhausting host RAM: each rank loads
    the full checkpoint, slices its expert shard, moves to GPU (freeing the
    CPU copy), then signals the next rank to proceed.

    Args:
        config: Training configuration.
        ep_group: EP process group (typically the world group).
        device: Target CUDA device for this rank.

    Returns:
        Model ready for EP training.
    """
    dtype = eval(config.dtype)  # noqa: S307 — controlled config value
    rank = dist.get_rank(ep_group)
    world_size = dist.get_world_size(ep_group)

    model_kwargs: dict = {"torch_dtype": dtype, "low_cpu_mem_usage": True}

    # Load sequentially so only one full checkpoint lives in CPU RAM at a time.
    # After apply_expert_parallel the full expert tensor is released; after
    # model.to(device) the remaining CPU tensors are freed.
    model = None
    for r in range(world_size):
        if rank == r:
            model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

            if config.gradient_checkpointing:
                model.gradient_checkpointing_enable()

            model = apply_expert_parallel(model, ep_group)

            lora_cfg = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)

            # Freeze everything except LoRA adapters. get_peft_model freezes
            # the base HF layers but ExpertParallelMLP params are raw
            # nn.Parameters outside PEFT's view, so freeze explicitly.
            for name, param in model.named_parameters():
                if "lora_" not in name:
                    param.requires_grad_(False)

            model.to(device)
            gc.collect()

        dist.barrier(group=ep_group)

    return model


_EXPERT_PARAM_NAMES = {
    "gate_proj",
    "up_proj",
    "gate_proj_bias",
    "up_proj_bias",
    "down_proj",
    "down_proj_bias",
}


def is_expert_param(name: str) -> bool:
    """Return True for parameters belonging to the local expert bank.

    Args:
        name: Fully qualified parameter name from model.named_parameters().

    Returns:
        True if the parameter is an expert weight (rank-local, no sync needed).
    """
    return name.split(".")[-1] in _EXPERT_PARAM_NAMES


def sync_replicated_gradients(model: nn.Module, ep_group: dist.ProcessGroup) -> None:
    """All-reduce gradients for every replicated (non-expert) parameter.

    Expert weights are rank-local and require no synchronisation.  All
    other parameters (attention, norms, embeddings, router, LoRA adapters)
    are replicated and must be averaged across ranks after each backward.

    Args:
        model: The EP model after loss.backward().
        ep_group: The EP process group to reduce over.
    """
    for name, param in model.named_parameters():
        if not param.requires_grad or is_expert_param(name):
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=ep_group)
