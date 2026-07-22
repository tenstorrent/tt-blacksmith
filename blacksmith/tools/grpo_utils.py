# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GRPO (Group Relative Policy Optimization) utilities.

Based on the paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning
in Open Language Models" https://arxiv.org/pdf/2402.03300

The GRPO objective (eq. 20) maximized here is, per completion token t of sample i:

    min( rho_{i,t} * A_i, clip(rho_{i,t}, 1-eps, 1+eps) * A_i ) - beta * KL_{i,t}

where:
    - rho_{i,t} = pi_theta(o_{i,t}|.) / pi_old(o_{i,t}|.)      (importance ratio)
    - A_i       = (r_i - mean(group)) / (std(group) + eps)     (group advantage)
    - KL_{i,t}  is the DeepSeekMath unbiased (k3) estimator of KL[pi_theta || pi_ref]:
                     pi_ref/pi_theta - log(pi_ref/pi_theta) - 1

``pi_old`` is a frozen weight copy of the policy used to sample completions. It is
synced from ``pi_theta`` once per batch; ``num_iterations`` (μ) policy updates then
reuse the same rollouts, rewards, and ``pi_old`` log-probs so only the importance
ratio changes. ``pi_ref`` is a separate frozen reference (base model for LoRA) used
only for KL.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def compute_group_advantages(
    rewards: torch.Tensor,
    num_prompts: int,
    num_generations: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Group-relative advantage normalization.

    ``rewards`` is a flat tensor of shape ``(num_prompts * num_generations,)``
    ordered so each consecutive ``num_generations`` block is one prompt's group.
    Returns a flat tensor of the same shape with per-group standardized rewards.
    """
    grouped = rewards.view(num_prompts, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantages = (grouped - mean) / (std + eps)
    return advantages.reshape(-1)


def get_per_token_logps(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log-probs of the realized next tokens.

    ``logits`` (B, T, V) predict ``input_ids`` shifted by one, so the returned
    tensor has shape (B, T-1): entry ``[b, t]`` is ``log pi(input_ids[b, t+1])``.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()
    batch_size, seq_len, vocab_size = shift_logits.shape
    per_token_logps = -F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        shift_targets.reshape(-1),
        reduction="none",
    ).reshape(batch_size, seq_len)
    return per_token_logps


def forward_logps(model, seq_ids: torch.Tensor, seq_attention_mask: torch.Tensor) -> torch.Tensor:
    """Run one full-sequence forward and return per-token log-probs (B, T-1).

    The (B, T, V) logits are dropped immediately after reduction to keep only the
    small per-token tensor (avoids holding a vocab-sized activation).
    """
    logits = model(input_ids=seq_ids, attention_mask=seq_attention_mask).logits
    return get_per_token_logps(logits, seq_ids)


def sync_old_policy(policy_model: torch.nn.Module, old_policy_model: torch.nn.Module) -> None:
    """Copy ``pi_theta`` weights into the frozen ``pi_old`` behavior policy."""
    old_policy_model.load_state_dict(policy_model.state_dict())


def compute_ref_logps(
    policy_model,
    seq_ids: torch.Tensor,
    seq_attention_mask: torch.Tensor,
    *,
    use_shared_reference: bool = True,
    reference_model=None,
) -> torch.Tensor:
    """Frozen reference-policy per-token log-probs (no grad).

    For LoRA, ``use_shared_reference=True`` runs the policy with adapters disabled
    (base weights = pi_ref) so a second full model copy is not needed. Otherwise
    ``reference_model`` must be a separate frozen model.
    """
    with torch.no_grad():
        if use_shared_reference:
            with policy_model.disable_adapter():
                return forward_logps(policy_model, seq_ids, seq_attention_mask)
        return forward_logps(reference_model, seq_ids, seq_attention_mask)


def compute_old_logps(
    old_policy_model,
    seq_ids: torch.Tensor,
    seq_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Frozen behavior-policy (pi_old) per-token log-probs (no grad).

    ``old_policy_model`` is the weight copy used to sample the completions, so these
    log-probs match the sampling distribution in the importance ratio
    ``pi_theta / pi_old``.
    """
    with torch.no_grad():
        return forward_logps(old_policy_model, seq_ids, seq_attention_mask)


def compute_grpo_loss(
    logps: torch.Tensor,
    ref_logps: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    beta: float = 0.005,
    epsilon: float = 0.2,
    old_logps: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the token-averaged GRPO loss (negative of the objective).

    Args:
        logps: Policy (pi_theta) per-token log-probs (B, T-1), gradient-enabled.
        ref_logps: Reference (pi_ref) per-token log-probs (B, T-1), detached.
        completion_mask: (B, T-1) float/bool mask, 1 for completion tokens.
        advantages: (B,) per-sample group advantages.
        beta: KL penalty coefficient.
        epsilon: PPO clip range.
        old_logps: (B, T-1) behavior-policy (pi_old) log-probs from the frozen
            weight copy that sampled the completions. If omitted, falls back to
            ``logps.detach()`` (ratio is 1 in value, clip inactive).

    Returns:
        (loss, metrics) where metrics holds detached ``loss``, ``kl``, and
        ``clip_frac`` (fraction of completion tokens where clipping is active).
    """
    completion_mask = completion_mask.to(logps.dtype)
    if old_logps is None:
        old_logps = logps.detach()

    # Importance ratio pi_theta / pi_old.
    ratio = torch.exp(logps - old_logps)
    advantages = advantages.unsqueeze(1)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
    policy_term = torch.min(unclipped, clipped)

    # DeepSeekMath k3 KL estimator: KL[pi_theta || pi_ref].
    log_ratio_ref = ref_logps - logps
    per_token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

    per_token_loss = -(policy_term - beta * per_token_kl)

    token_counts = completion_mask.sum(dim=1).clamp(min=1.0)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / token_counts).mean()
    kl = ((per_token_kl.detach() * completion_mask).sum(dim=1) / token_counts).mean()

    clipped_mask = (ratio < (1.0 - epsilon)) | (ratio > (1.0 + epsilon))
    clip_frac = (clipped_mask.to(logps.dtype) * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

    return loss, {"loss": loss.detach(), "kl": kl, "clip_frac": clip_frac.detach()}
