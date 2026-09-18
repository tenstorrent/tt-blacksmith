# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


def negative_log_likelihood_loss(
    shift_logits: torch.Tensor,
    expected_output: torch.Tensor,
    loss_scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Token-averaged causal-LM loss written for the ops tt-crank lowers.

    `loss_scale` is `-mask / valid_tokens`, pre-divided on the host so the graph needs no
    `aten.div`. The final sum takes an explicit dim list because only `aten.sum.dim_IntList`
    is lowered on tt, not the full-reduction overload, which is also why the result is a
    `[1, 1, 1]` tensor rather than a 0-d scalar.

    Args:
        shift_logits: `[batch, seq_len - 1, vocab]` logits for the next-token positions.
        expected_output: One-hot targets of the same shape (zeros on masked positions).
        loss_scale: `[batch, seq_len - 1, 1]` per-token weights, `-1 / valid_tokens` or 0.
        dtype: Compute dtype the log-probabilities are cast back to.

    Returns:
        The `[1, 1, 1]` loss.
    """
    probabilities = torch.softmax(shift_logits, dim=-1)
    target_probabilities = (probabilities * expected_output).sum(dim=-1, keepdim=True)
    log_probabilities = torch.log(target_probabilities.float().clamp_min(1e-12)).to(dtype)
    return (log_probabilities * loss_scale).sum(dim=[0, 1, 2], keepdim=True)
