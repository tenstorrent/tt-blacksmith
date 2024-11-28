# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import wandb
import torch
import numpy as np


def log_histogram(experiment, name, tensor, step, n_bins=100):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach().numpy()

    hist, bins = np.histogram(tensor, bins=n_bins)

    experiment._log(
        {
            name: wandb.Histogram(np_histogram=(hist, bins)),
        },
        step=step,
    )
