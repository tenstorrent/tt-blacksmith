# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from blacksmith.datasets.torch.gsm8k.gsm8k_dataset import (
    DATASET_PATH,
    SYSTEM_PROMPT,
    GSM8KDataset,
)

__all__ = [
    "GSM8KDataset",
    "SYSTEM_PROMPT",
    "DATASET_PATH",
]
