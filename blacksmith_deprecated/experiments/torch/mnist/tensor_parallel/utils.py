# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F


def cross_entropy_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Workaround for nn.CrossEntropyLoss - it returns a scalar (reduction='mean'),
    # but tensor parallel operations require loss shape [1, 1] (keepdim=True).
    # github issue: https://github.com/tenstorrent/tt-xla/issues/1993
    if targets.dim() == 2 and targets.size(1) == outputs.size(1):
        log_probs = F.log_softmax(outputs, dim=1)
        per_sample = -(log_probs * targets).sum(dim=1, keepdim=True)
    else:
        per_sample = F.cross_entropy(outputs, targets, reduction="none").unsqueeze(1)
    return per_sample.mean(dim=0, keepdim=True)
