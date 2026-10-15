# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.distributed.tensor import DTensor

TORCH_DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}


def torch_dtype(dtype: str) -> torch.dtype:
    """Map a config dtype string (e.g. ``"torch.bfloat16"``) to a ``torch.dtype``."""
    try:
        return TORCH_DTYPES[dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported dtype {dtype!r}; expected one of {sorted(TORCH_DTYPES)}") from e


def to_host(obj):
    """Move tensors in a (nested) container to CPU, gathering DTensor shards."""
    if isinstance(obj, DTensor):
        return obj.full_tensor().cpu()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_host(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_host(v) for v in obj)
    return obj


def loss_to_float(loss: torch.Tensor | float) -> float:
    """Pull a loss to host as a float, reducing a `Partial` DTensor first."""
    if not isinstance(loss, torch.Tensor):
        return float(loss)
    if isinstance(loss, DTensor):
        loss = loss.full_tensor()
    return float(loss.detach().cpu())


def collate_fn_for_causal_lm(batch):
    """
    Collate function that pre-shifts labels for Causal LM.
    Shifts labels to exclude first token.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    shifted_labels = labels[:, 1:].contiguous()

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": shifted_labels}
