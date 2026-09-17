# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import torch

from blacksmith.tools.templates.configs import TrainingConfig


class ReproducibilityManager:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def setup(self):
        self._seed_python_rngs()

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _seed_python_rngs(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
