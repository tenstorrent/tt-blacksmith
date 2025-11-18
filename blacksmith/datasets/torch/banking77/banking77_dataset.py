# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.experiments.torch.llama.configs import TrainingConfig


class Banking77Dataset(BaseDataset):
    def __init__(self, config: TrainingConfig, split: str = "train"):
        """
        Args:
            config: Training configuration
            split: Dataset split to use ("train" or "test")
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, padding_side="right", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.required_columns = ["input_ids", "attention_mask", "labels"]
        self.split = split
        self.label_map = None

        self._prepare_dataset()

    def _tokenize_function(self, example: Dict) -> Dict:
        encoding = self.tokenizer(example["text"], truncation=False, padding=False, return_tensors="pt")

        example["input_ids"] = encoding["input_ids"].squeeze(0)
        example["attention_mask"] = encoding["attention_mask"].squeeze(0)
        example["labels"] = example["label"]

        return example

    def _prepare_dataset(self):
        self.raw_dataset = load_dataset(self.config.dataset_id, split=self.split)

        # Create label mapping
        unique_labels = sorted(set(self.raw_dataset["label"]))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_labels = len(self.label_map)

        self.dataset = self.raw_dataset.map(self._tokenize_function)
        self.dataset = self.dataset.remove_columns(
            [col for col in self.dataset.column_names if col not in self.required_columns]
        )

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx]

    def _collate_fn(self, batch):
        input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
        attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
        labels = torch.stack([torch.tensor(item["labels"]) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def get_dataloader(self) -> DataLoader:
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding=True, max_length=self.config.max_length
        )

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=self.split == "train",
            drop_last=True,
        )

        return dataloader

    def get_num_labels(self) -> int:
        return self.num_labels

    def get_label_map(self) -> Dict:
        return self.label_map
