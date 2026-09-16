# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal checkpointing: save trainable (LoRA) weights to host."""
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from blacksmith.tools.configs import TrainingConfig
from blacksmith.tools.logging_manager import TrainingLogger


class CheckpointManager:
    def __init__(self, config: TrainingConfig, logger: TrainingLogger):
        self.config = config
        self.logger = logger
        self.output_dir = Path(config.output_dir)
        self._saved: list[Path] = []

    def should_save(self, global_step: int, epoch_end: bool = False) -> bool:
        if self.config.save_strategy == "step":
            return global_step > 0 and global_step % self.config.steps_freq == 0
        if self.config.save_strategy == "epoch":
            return epoch_end
        return False

    def save(self, model: nn.Module, global_step: int, name: Optional[str] = None) -> Path:
        """Write the trainable parameters to disk.

        Values are pulled to host with `.cpu()`; under tensor parallelism a
        sharded parameter is a DTensor, so `full_tensor()` gathers it first --
        the checkpoint is always the unsharded, host-side view.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / (name or f"checkpoint_step_{global_step}.pth")

        state = {}
        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            value = param.detach()
            if isinstance(value, DTensor):
                value = value.full_tensor()
            state[param_name] = value.cpu()

        torch.save({"step": global_step, "state_dict": state}, path)
        self.logger.info(f"Saved checkpoint to {path}")

        self._saved.append(path)
        while len(self._saved) > self.config.keep_last_n:
            stale = self._saved.pop(0)
            stale.unlink(missing_ok=True)

        return path
