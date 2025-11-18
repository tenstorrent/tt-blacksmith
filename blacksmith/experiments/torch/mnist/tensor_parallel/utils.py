# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_xla
import torch_xla.distributed.spmd as xs


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


def apply_tensor_parallel_sharding(model: nn.Module, mesh: xs.Mesh):
    # Sync to ensure all weights are materialized
    torch_xla.sync(wait=True)
    # Layer names and their corresponding sharding patterns
    layer_configs = {
        "linear_relu_stack.0": (None, "model"),
        "linear_relu_stack.2": ("model", None),
        "linear_relu_stack.4": (None, "model"),
    }

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, "weight"):
            if name in layer_configs:
                sharding_pattern = layer_configs[name]
                xs.mark_sharding(module.weight, mesh, sharding_pattern)
