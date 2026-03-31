# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch.distributed as dist


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialise NCCL process group and return rank and device info.

    Returns:
        Tuple of (global rank, local rank, CUDA device for this rank).
    """
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("LOCAL_RANK not set. Launch with torchrun or set LOCAL_RANK manually.")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    return rank, local_rank, device


def is_main_process() -> bool:
    """Return True if this is the rank-0 process."""
    return dist.get_rank() == 0
