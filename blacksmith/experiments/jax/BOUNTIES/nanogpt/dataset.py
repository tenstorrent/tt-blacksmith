# SPDX-FileCopyrightText: (c) 2022 Andrej Karpathy
#
# SPDX-License-Identifier: MIT

import numpy as np

from datasets import load_dataset


class ShakespeareDataset:
    def __init__(self):

        # 1. Download dataset.
        print("Loading `tiny_shakespeare` from Hugging Face...")
        # Latest update from datasets forbids loading from alias like "karpathy/tiny_shakespeare"
        # because of the potential security breaches, instead it's required to use raw URL.
        ds = load_dataset(
            "text",
            data_files={
                "train": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            },
        )

        # 2. Merge data.
        # Please consider that we do not account for unknown tokens.
        full_text = "".join(ds["train"]["text"])

        # 3. Build Vocabulary.
        self.chars = sorted(list(set(full_text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # 4. Tokenize.
        print(f"Tokenizing {len(full_text):,} characters...")
        full_ids = np.array([self.stoi[c] for c in full_text], dtype=np.uint32)

        # 5. Split by Karpathy standard.
        split_idx = int(len(full_ids) * 0.9)
        self.train_data = full_ids[:split_idx]
        self.val_data = full_ids[split_idx:]

        print(f"Vocab size: {self.vocab_size}")
        print(f"Train tokens: {len(self.train_data):,}")
        print(f"Val tokens:   {len(self.val_data):,}")

    def get_data(self, split):
        return self.train_data if split == "train" else self.val_data

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join([self.itos[int(i)] for i in ids])
