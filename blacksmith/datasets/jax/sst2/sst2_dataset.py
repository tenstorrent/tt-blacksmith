# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np

from blacksmith.datasets.torch.sst2.sst2_dataset import SSTDataset
from blacksmith.tools.templates.configs import TrainingConfig

logger = logging.getLogger(__name__)


def _create_batches(data: np.ndarray, batch_size: int) -> np.ndarray:
    # Reshape flat numpy data into batches of shape (num_batches, batch_size, seq_len).
    num_batches = len(data) // batch_size
    return data[: num_batches * batch_size].reshape(num_batches, batch_size, -1)


def load_sst2_batches(
    config: TrainingConfig,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SST-2 instruction-CLM batches via the torch SSTDataset.

    Returns:
        (input_ids, labels, attention_masks), each a numpy array of shape
        (num_batches, batch_size, seq_len).
    """
    dataset = SSTDataset(config, split=split)
    dataloader = dataset.get_dataloader()

    all_input_ids: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_attention_masks: list[np.ndarray] = []
    for batch in dataloader:
        for input_ids_tensor in batch["input_ids"]:
            all_input_ids.append(np.array(input_ids_tensor))
        for labels_tensor in batch["labels"]:
            all_labels.append(np.array(labels_tensor))
        for attention_mask_tensor in batch["attention_mask"]:
            all_attention_masks.append(np.array(attention_mask_tensor))

    input_ids = _create_batches(np.stack(all_input_ids).astype(np.uint32), config.batch_size)
    labels = _create_batches(np.stack(all_labels).astype(np.int32), config.batch_size)
    attention_masks = _create_batches(np.stack(all_attention_masks).astype(np.int32), config.batch_size)

    logger.info(
        f"  prepared {len(input_ids)} {split} SST-2 batches " f"of shape ({config.batch_size}, {config.max_length})"
    )
    return input_ids, labels, attention_masks
