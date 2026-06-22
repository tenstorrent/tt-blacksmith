# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DPO (Direct Preference Optimization) utilities.

Based on the paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/pdf/2305.18290

The DPO loss is:
    L_DPO = -log(sigmoid(beta * (log_pi(y_w|x) - log_pi(y_l|x) - log_pi_ref(y_w|x) + log_pi_ref(y_l|x))))

Where:
    - y_w is the chosen (winning) response
    - y_l is the rejected (losing) response
    - pi is the policy model being trained
    - pi_ref is the reference model (frozen)
    - beta is the temperature parameter
"""
import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORED_LABEL_ID = -100


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_prob: bool = True,
) -> torch.Tensor:
    """
    Compute log probabilities for the supervised label tokens.

    Args:
        logits: Model output (batch_size, seq_len, vocab_size)
        labels: Target token IDs (batch_size, seq_len), -100 for ignored positions
        average_log_prob: If True, return the mean log prob per supervised token.
            If False, return the summed log prob over the sequence. Averaging avoids
            length bias when chosen and rejected responses differ in length.

    Returns:
        log_probs: (batch_size,) log prob per sequence
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"Logits shape {logits.shape[:-1]} doesn't match labels shape {labels.shape}")

    # Shift for causal LM: logits[i] predicts labels[i+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    batch_size, seq_len, vocab_size = shift_logits.shape

    labels_mask = shift_labels != IGNORED_LABEL_ID
    # cross_entropy would index out of range on the ignored sentinel (-100), so point those
    # positions at a valid class; their contribution is removed by labels_mask below.
    safe_labels = torch.where(labels_mask, shift_labels, torch.zeros_like(shift_labels))

    # F.cross_entropy(reduction="none") returns -log p(label) per token via an internal gather,
    # avoiding a [seq_len, vocab_size] one-hot tensor (which tile-pads to multi-GB on TT and OOMs).
    per_token_logps = -F.cross_entropy(
        shift_logits.reshape(-1, vocab_size),
        safe_labels.reshape(-1),
        reduction="none",
    ).reshape(batch_size, seq_len)
    per_token_logps = per_token_logps * labels_mask.float()

    if average_log_prob:
        return per_token_logps.sum(dim=-1) / labels_mask.sum(dim=-1).float().clamp(min=1.0)
    return per_token_logps.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the DPO loss for a batch of preference pairs.

    Args:
        policy_chosen_logps: Log probs of chosen responses under policy model
        policy_rejected_logps: Log probs of rejected responses under policy model
        reference_chosen_logps: Log probs of chosen responses under reference model
        reference_rejected_logps: Log probs of rejected responses under reference model
        beta: Temperature parameter (higher = more conservative updates)
        label_smoothing: Label smoothing parameter (0 = no smoothing)

    Returns:
        Tuple of (loss, chosen_rewards, rejected_rewards). Reward tensors are per-sample;
        callers should reduce them before logging if needed.
    """
    # Compute log ratios
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # DPO implicit reward: beta * (log_pi(y|x) - log_pi_ref(y|x))
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    # DPO loss: -log(sigmoid(beta * (pi_logratio - ref_logratio)))
    logits = beta * (pi_logratios - ref_logratios)

    if label_smoothing > 0:
        # Soft labels for label smoothing
        losses = -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
    else:
        losses = -F.logsigmoid(logits)

    loss = losses.mean()

    return loss, chosen_rewards, rejected_rewards


def create_reference_model(model: nn.Module) -> nn.Module:
    """
    Create a frozen copy of the model to use as the reference model for DPO.

    Args:
        model: The policy model

    Returns:
        A frozen copy of the model
    """
    reference_model = copy.deepcopy(model)

    # Freeze all parameters
    for param in reference_model.parameters():
        param.requires_grad = False

    reference_model.eval()

    return reference_model


def _forward_batch_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Run a forward pass, compute sequence log-probs, then drop logits before the next pass."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    logps = get_batch_logps(logits, labels)
    return logps


def compute_dpo_loss_from_batch(
    policy_model: nn.Module,
    reference_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute DPO loss for a batch of preference pairs.

    Args:
        policy_model: The model being trained
        reference_model: Frozen reference model
        batch: Batch containing chosen and rejected sequences
        beta: DPO temperature parameter
        label_smoothing: Label smoothing parameter

    Returns:
        Tuple of (loss, metrics_dict)
    """
    with torch.no_grad():
        reference_chosen_logps = _forward_batch_logps(
            reference_model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        ).detach()

        reference_rejected_logps = _forward_batch_logps(
            reference_model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        ).detach()

    policy_chosen_logps = _forward_batch_logps(
        policy_model,
        batch["chosen_input_ids"],
        batch["chosen_attention_mask"],
        batch["chosen_labels"],
    )

    policy_rejected_logps = _forward_batch_logps(
        policy_model,
        batch["rejected_input_ids"],
        batch["rejected_attention_mask"],
        batch["rejected_labels"],
    )

    # Compute DPO loss
    loss, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=beta,
        label_smoothing=label_smoothing,
    )

    # Per-sample accuracy: how often chosen has higher implicit reward than rejected
    reward_margin = chosen_rewards - rejected_rewards
    accuracy = (reward_margin > 0).float().mean()

    # Log-prob drift from reference (unscaled); growing values mean policy is diverging from pi_ref
    kl_chosen = (policy_chosen_logps - reference_chosen_logps).mean()
    kl_rejected = (policy_rejected_logps - reference_rejected_logps).mean()

    metrics = {
        "loss": loss,
        "chosen_rewards": chosen_rewards.mean(),
        "rejected_rewards": rejected_rewards.mean(),
        "reward_margin": reward_margin.mean(),
        "accuracy": accuracy,
        "kl_chosen": kl_chosen,
        "kl_rejected": kl_rejected,
    }

    return loss, metrics
