# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from blacksmith.datasets.jax.alpaca.alpaca_dataset import load_alpaca_batches
from blacksmith.datasets.jax.sst2.sst2_dataset import load_sst2_batches
from blacksmith.tools.templates.configs import TrainingConfig

AVAILABLE_DATASETS = {
    "sst2": load_sst2_batches,
    "alpaca": load_alpaca_batches,
}


def load_batches(
    config: TrainingConfig,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset batches based on config.dataset_id.

    Returns:
        (input_ids, labels, attention_masks), each a numpy array of shape
        (num_batches, batch_size, seq_len).
    """
    loader = AVAILABLE_DATASETS.get(config.dataset_id.lower())
    if loader is None:
        raise ValueError(
            f"Unknown JAX dataset: {config.dataset_id!r}. "
            f"Available: {sorted(AVAILABLE_DATASETS)}"
        )
    return loader(config, split)
