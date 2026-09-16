# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from datasets import load_dataset


def load_sst2(tokenizer, max_length=128):
    raw = load_dataset("glue", "sst2")

    def tokenize_batch(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = raw.map(tokenize_batch, batched=True, remove_columns=["sentence"])
    tokenized = tokenized.rename_column("label", "labels")
    columns = ["input_ids", "attention_mask", "labels"]
    tokenized.set_format(type="np", columns=columns)

    return tokenized["train"], tokenized["validation"], columns


# Batch iterator that shuffles dataset order at the start of each epoch.
def numpy_batch_iter(dataset, batch_size, columns, shuffle=True, seed=None):
    n = len(dataset)
    rng = np.random.default_rng(seed)
    order = np.arange(n)

    while True:
        if shuffle:
            rng.shuffle(order)

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch = {k: dataset[k][idx] for k in columns}
            batch["input_ids"] = batch["input_ids"].astype(np.int32)
            batch["attention_mask"] = batch["attention_mask"].astype(np.int32)
            batch["labels"] = batch["labels"].astype(np.int32)

            yield batch
