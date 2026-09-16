# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import random

import numpy as np
import torch

from blacksmith.tools.configs import TrainingConfig


class ReproducibilityManager:
    """Seeds host RNGs.

    Only host-side randomness is covered: dataset shuffling, dropout masks and
    weight init all draw from the CPU generator before anything reaches the
    device, and tt-crank has no separately seedable device RNG.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

    def setup(self) -> None:
        seed = self.config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
