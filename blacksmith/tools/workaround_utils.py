# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F


# Custom cross-entropy loss because of https://github.com/tenstorrent/tt-xla/issues/1993.
def cross_entropy_loss(shift_logits, expected_output, labels_mask):
    log_probs = F.log_softmax(shift_logits, dim=-1)  # [batch, seq_len, vocab_size]
    # Cross entropy: -sum(target * log_prob) over vocab dimension
    ce_loss = -(expected_output * log_probs).sum(dim=-1, keepdim=True)  # [batch, seq_len, 1]

    # Apply mask to ignore padding tokens
    labels_mask = labels_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
    ce_loss = ce_loss * labels_mask

    # Compute mean only over valid (non-masked) positions
    num_valid = labels_mask.sum(dim=1, keepdim=True)  # [batch, 1, 1]
    num_valid = torch.clamp(num_valid, min=1.0)  # Avoid division by zero
    loss_per_sample = ce_loss.sum(dim=1, keepdim=True) / num_valid  # [batch, 1, 1]
    loss = loss_per_sample.mean(dim=0, keepdim=True)  # [1, 1, 1]
    return loss


# Used in conjunction with cross_entropy_loss.
def transform_labels(batch, ignored_index, vocab_size):
    labels = batch["labels"]
    labels_mask = labels != ignored_index
    labels[labels == ignored_index] = 0
    expected_output = F.one_hot(labels, num_classes=vocab_size)

    return expected_output, labels_mask
