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
        self._setup_python()

        if self.config.framework.lower() == "pytorch":
            self._setup_pytorch()
        elif self.config.framework.lower() == "jax":
            self._setup_jax()
        else:
            print(f"Unknown framework: {self.config.framework}")

    def _setup_python(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _setup_pytorch(self):
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _setup_jax(self):
        pass
