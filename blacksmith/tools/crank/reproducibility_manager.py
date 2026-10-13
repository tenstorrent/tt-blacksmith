# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

from blacksmith.tools.reproducibility_manager import ReproducibilityManager


class CrankReproducibilityManager(ReproducibilityManager):
    """Adds torch / CUDA / cuDNN seeding on top of the framework-agnostic base."""

    def setup(self):
        super().setup()

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
