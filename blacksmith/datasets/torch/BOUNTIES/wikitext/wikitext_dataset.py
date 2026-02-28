# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wikitext-2 Dataset Implementation for Causal Language Model Training.

This module provides a dataset wrapper for the Wikitext-2 dataset,
suitable for causal language model fine-tuning.
"""
from typing import Dict, List

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig
from datasets import load_dataset

DATASET_BENCHMARK = "wikitext"
DATASET_NAME = "wikitext-2-raw-v1"


class WikitextDataset(BaseDataset):
    """
    Wikitext-2 dataset for causal language model training.

    Processes the Wikitext-2 dataset into tokenized chunks suitable
    for training LLMs with LoRA.
    """

    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="right", use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.required_columns = ["input_ids", "attention_mask", "labels"]
        super().__init__(config, split, collate_fn)

    def _prepare_dataset(self):
        """Load and prepare the Wikitext-2 dataset."""
        raw_dataset = load_dataset(DATASET_BENCHMARK, DATASET_NAME, split=self.split)

        # Filter out empty examples.
        raw_dataset = raw_dataset.filter(lambda example: len(example["text"].strip()) > 0)

        # Tokenize and chunk the dataset.
        tokenized_dataset = raw_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
            desc=f"Tokenizing {self.split} split",
        )

        # Group texts into chunks of max_length.
        self.dataset = tokenized_dataset.map(
            self._group_texts,
            batched=True,
            desc=f"Grouping texts into chunks of {self.config.max_length}",
        )

    def _tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize the text examples."""
        return self.tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            return_attention_mask=True,
        )

    def _group_texts(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Group tokenized texts into chunks of max_length.

        This concatenates all texts and then splits them into fixed-length
        chunks for efficient training.
        """
        # Concatenate all input_ids and attention_masks
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}

        total_length = len(concatenated["input_ids"])
        block_size = self.config.max_length

        # Drop the remainder that doesn't fit into a full block
        total_length = (total_length // block_size) * block_size

        # Split by chunks of block_size
        result = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for i in range(0, total_length, block_size):
            input_ids = concatenated["input_ids"][i : i + block_size]
            attention_mask = concatenated["attention_mask"][i : i + block_size]

            result["input_ids"].append(input_ids)
            result["attention_mask"].append(attention_mask)
            # Labels are same as input_ids for causal LM
            result["labels"].append(input_ids.copy())

        return result

    def __getitem__(self, idx: int) -> Dict:
        """Get a single example from the dataset."""
        sample = self.dataset[idx]

        import torch

        return {
            "input_ids": torch.tensor(sample["input_ids"]),
            "attention_mask": torch.tensor(sample["attention_mask"]),
            "labels": torch.tensor(sample["labels"]),
        }

    def _get_dataloader(self) -> DataLoader:
        """Create and return a DataLoader for this dataset."""
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8,  # For efficient tensor operations
        )

        if self.collate_fn is not None:

            def combined_collate_fn(batch):
                collated = data_collator(batch)
                return self.collate_fn(collated)

            collate_function = combined_collate_fn
        else:
            collate_function = data_collator

        batch_size = self.config.batch_size

        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_function,
            shuffle=(self.split == "train"),
            drop_last=(self.split == "train"),
            num_workers=0,
            pin_memory=False,
        )
