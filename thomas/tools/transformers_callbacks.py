# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import wandb
import numpy as np
from transformers import TrainerCallback


class WandbMemoryCallback(TrainerCallback):
    """Callback to log GPU memory usage to Weights & Biases."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            logs = logs or {}
            logs["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
            logs["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
            wandb.log(logs)


class GradientSavingCallback(TrainerCallback):
    """Callback to save model parameter gradients during training."""

    def __init__(self, gradients_dir):
        self.gradients = {}
        self.next_epoch = 0
        self.gradients_dir = gradients_dir

    def on_optimizer_step(self, args, state, control, model, **kwargs):
        if state.epoch < self.next_epoch:
            return control

        print(f"Saving gradients epoch {self.next_epoch}...")
        self.next_epoch += 1

        for name, param in model.named_parameters():
            if param.grad is not None:
                self.gradients[name] = param.grad.clone().detach().cpu().numpy()

        for name, grad in self.gradients.items():
            np.savetxt(os.path.join(self.gradients_dir, f"{name}_epoch_{state.epoch}.csv"), grad, delimiter=",")

        return control
