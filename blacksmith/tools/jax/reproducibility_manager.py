# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import random

import jax
import numpy as np

from blacksmith.tools.templates.configs import TrainingConfig


class JaxReproducibilityManager:
    """JAX/EasyDel counterpart to ReproducibilityManager."""

    def __init__(self, config: TrainingConfig):
        self.config = config

    def setup(self):
        self._seed_python_rngs()

        if self.config.deterministic:
            jax.config.update("jax_default_matmul_precision", "highest")

    def _seed_python_rngs(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

    def get_jax_rng(self):
        return jax.random.PRNGKey(self.config.seed)
