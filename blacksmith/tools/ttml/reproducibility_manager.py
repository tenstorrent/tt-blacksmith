# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import warnings

import ttml

from blacksmith.tools.reproducibility_manager import ReproducibilityManager


class TtmlReproducibilityManager(ReproducibilityManager):
    """Adds ttml seeding (parameter init and the autograd context RNG) on top of the framework-agnostic base."""

    def setup(self):
        # Python / numpy first: ttml's LoRA `lora_A` init draws from numpy's global RNG.
        super().setup()
        ttml.manual_seed(self.config.seed)

        if self.config.deterministic:
            warnings.warn("`deterministic: True` has no ttml equivalent; seeding is all that is applied.")
