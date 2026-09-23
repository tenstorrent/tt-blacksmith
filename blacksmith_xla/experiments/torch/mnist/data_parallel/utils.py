# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


def mse_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Workaround for nn.MSELoss - it returns a scalar (reduction='mean'),
    # but data parallel operations require loss shape [1, 1] (keepdim=True).
    # github issue: https://github.com/tenstorrent/tt-xla/issues/1993
    loss = (outputs - targets).pow(2)
    loss = loss.mean(dim=1, keepdim=True)
    loss = loss.mean(dim=0, keepdim=True)
    return loss
