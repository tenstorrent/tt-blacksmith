# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

import torch

from blacksmith.tools.checkpoints_manager import (
    CheckpointManager as XlaCheckpointManager,
)
from blacksmith.tools.crank.torch_helpers import to_host


class _HostStateView:
    """Proxy whose `state_dict()` is the wrapped object's, copied to the host."""

    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def state_dict(self) -> Dict[str, Any]:
        return to_host(self._obj.state_dict())


class CheckpointManager(XlaCheckpointManager):
    """CheckpointManager for tt-crank.

    Identical to the shared one except that model and optimizer state are copied to
    the host before `torch.save`: a `tt` tensor cannot be pickled in place, and under
    tensor parallelism the parameters are DTensor shards that have to be gathered.
    Loading is inherited unchanged (`restore_capturable_optimizer_state` is a no-op
    off XLA).
    """

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        step: int,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        return super().save_checkpoint(
            _HostStateView(model),
            step,
            epoch,
            _HostStateView(optimizer) if optimizer is not None else None,
            metrics,
            checkpoint_name,
        )
