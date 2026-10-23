# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttml host batches: layout, and the same loss as crank's ignore_index cross-entropy."""
import numpy as np
import pytest

from blacksmith.tools.ttml.collate import IGNORED_INDEX, collate_causal_lm

PAD = 7
EXAMPLES = [
    # 2 prompt tokens masked out of the labels, then 3 response tokens.
    {"input_ids": [11, 12, 13, 14, 15], "labels": [IGNORED_INDEX, IGNORED_INDEX, 13, 14, 15]},
    {"input_ids": [21, 22, 23], "labels": [IGNORED_INDEX, 22, 23]},
]


def _cross_entropy(logits, targets):
    """Per-position CE, float64. logits [..., V], targets [...]."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return -np.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]


def test_layout_padding_and_shift():
    batch = collate_causal_lm(EXAMPLES, max_length=8, pad_token_id=PAD)
    assert batch["input_ids"].shape == (2, 1, 1, 8) and batch["input_ids"].dtype == np.uint32
    assert batch["labels"].shape == (2, 8) and batch["labels"].dtype == np.uint32
    assert batch["loss_mask"].shape == (2, 1, 8, 1)

    assert batch["input_ids"][0, 0, 0].tolist() == [11, 12, 13, 14, 15, PAD, PAD, PAD]
    # Position t predicts token t + 1; positions without a target hold the pad id.
    assert batch["labels"][0].tolist() == [PAD, 13, 14, 15, PAD, PAD, PAD, PAD]
    assert batch["labels"][1].tolist() == [22, 23, PAD, PAD, PAD, PAD, PAD, PAD]

    mask = batch["loss_mask"][:, 0, :, 0]
    assert (mask[0] > 0).tolist() == [False, True, True, True, False, False, False, False]
    # Normalized so mean(CE * mask) is the mean over the 5 real targets.
    assert mask.sum() == pytest.approx(2 * 8)


def test_loss_equals_crank_ignore_index_cross_entropy():
    rng = np.random.default_rng(0)
    batch_size, max_length, vocab = 2, 8, 32
    logits = rng.normal(size=(batch_size, max_length, vocab))

    # crank: DataCollatorForSeq2Seq pads labels with -100, labels[:, 1:] vs logits[:, :-1].
    labels = np.full((batch_size, max_length), IGNORED_INDEX)
    for row, example in enumerate(EXAMPLES):
        labels[row, : len(example["labels"])] = example["labels"]
    shifted, valid = labels[:, 1:], labels[:, 1:] != IGNORED_INDEX
    crank_loss = _cross_entropy(logits[:, :-1], np.where(valid, shifted, 0))[valid].mean()

    batch = collate_causal_lm(EXAMPLES, max_length=max_length, pad_token_id=PAD)
    ttml_loss = (_cross_entropy(logits, batch["labels"].astype(np.int64)) * batch["loss_mask"][:, 0, :, 0]).mean()

    # Exact up to the float32 mask.
    assert ttml_loss == pytest.approx(crank_loss, rel=1e-6)


def test_too_long_example_raises():
    with pytest.raises(ValueError, match="max_length"):
        collate_causal_lm(EXAMPLES, max_length=4, pad_token_id=PAD)
