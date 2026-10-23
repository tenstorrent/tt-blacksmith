# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from functools import partial
from typing import Any, Optional

import numpy as np
import ttml
import ttnn
from ttml.datasets import InMemoryDataloader

from blacksmith.tools.ttml.collate import collate_causal_lm

# Device layout and dtype of each key `collate_causal_lm` produces.
BATCH_SPEC = {
    "input_ids": (ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32),
    "labels": (ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32),
    "loss_mask": (ttnn.Layout.TILE, ttnn.DataType.BFLOAT16),
}


def to_ttml(key: str, array: np.ndarray) -> Any:
    """Upload one host batch entry with the layout / dtype its key needs on device."""
    layout, dtype = BATCH_SPEC[key]
    return ttml.autograd.Tensor.from_numpy(array, layout, dtype)


def build_dataloader(dataset: Any, config: Any) -> Optional[InMemoryDataloader]:
    """A ttml dataloader over a blacksmith ``LLMDataset``, yielding host batches.

    ``None`` when the split is not configured (e.g. a custom dataset without ``val_dataset_path``).
    """
    if dataset.dataset is None:
        return None
    collate = partial(collate_causal_lm, max_length=config.max_length, pad_token_id=dataset.tokenizer.pad_token_id)
    return InMemoryDataloader(
        dataset,
        collate,
        batch_size=config.batch_size,
        shuffle=dataset.split == "train",
        drop_last=True,
        seed=config.seed,
    )
