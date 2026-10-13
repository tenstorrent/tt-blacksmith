# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np

from blacksmith.configs.trainer import TrainerConfig


class ReproducibilityManager:
    """Framework-agnostic seeding."""

    def __init__(self, config: TrainerConfig):
        self.config = config

    def setup(self):
        self._seed_python_rngs()

    def _seed_python_rngs(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
