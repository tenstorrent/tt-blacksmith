# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.distributed.tensor import DTensor


def to_host(obj):
    """Recursively move tensors in a (nested) container to CPU, gathering DTensor shards.

    tt-crank tensors cannot be serialized in place (`torch.save` tries to rebind a
    `tt` storage to a CPU tensor), and a sharded DTensor has to be gathered first
    so a checkpoint is always the unsharded, host-side view.
    """
    if isinstance(obj, DTensor):
        return obj.full_tensor().cpu()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_host(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_host(v) for v in obj)
    return obj


def loss_to_float(loss: torch.Tensor) -> float:
    """Pull a loss value to host as a plain float.

    Under data parallelism the loss reduces over the sharded batch dim, so it
    comes back as a `Partial` DTensor -- each chip holds a piece of the sum.
    `full_tensor()` fires the all-reduce that makes it the real value;
    `.item()` alone would silently report one chip's partial.
    """
    if isinstance(loss, DTensor):
        loss = loss.full_tensor()
    return float(loss.detach().cpu())
