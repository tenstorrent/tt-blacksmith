# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np

from blacksmith.experiments.easydel.qwen.configs import TrainingConfig

logger = logging.getLogger(__name__)


def create_batches(data: np.ndarray, batch_size: int = 4) -> np.ndarray:
    """Reshape flat numpy data into batches, dropping remainder.

    Args:
        data: Array of shape (num_examples, seq_length).
        batch_size: Number of samples per batch.

    Returns:
        Array of shape (num_batches, batch_size, seq_length).

    """
    num_batches = len(data) // batch_size
    return data[: num_batches * batch_size].reshape(num_batches, batch_size, -1)


def load_sst2_batches(
    config: TrainingConfig,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SST-2 instruction-CLM batches via SSTDataset.

    Uses the same prompt/response templates as the Torch SST
    experiments (blacksmith.datasets.torch.sst2). Labels contain -100
    at prompt positions so only response tokens contribute to the loss.

    Returns:
        Tuple of (input_ids, labels, attention_mask), each a numpy
        array of shape (num_batches, batch_size, seq_len).
    """
    from blacksmith.datasets.torch.sst2.sst2_dataset import SSTDataset

    dataset = SSTDataset(config, split=split)
    dataloader = dataset.get_dataloader()

    all_ids, all_labels, all_masks = [], [], []
    for batch in dataloader:
        for item in batch["input_ids"]:
            all_ids.append(np.array(item))
        for item in batch["labels"]:
            all_labels.append(np.array(item))
        for item in batch["attention_mask"]:
            all_masks.append(np.array(item))

    ids = create_batches(np.stack(all_ids).astype(np.uint32), config.batch_size)
    labels = create_batches(np.stack(all_labels).astype(np.int32), config.batch_size)
    masks = create_batches(np.stack(all_masks).astype(np.int32), config.batch_size)

    logger.info(f"  prepared {len(ids)} {split} SST-2 batches " f"of shape ({config.batch_size}, {config.max_length})")
    return ids, labels, masks
