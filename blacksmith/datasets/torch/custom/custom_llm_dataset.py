# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from blacksmith.datasets.torch.custom.custom_dataset_utils import (
    build_prompt,
    normalize_file_type,
    resolve_column_mapping,
)
from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.trainer.configs import TrainerConfig
from datasets import load_dataset


class CustomLLMDataset(BaseDataset):
    def __init__(self, config: TrainerConfig, split: str = "train", collate_fn=None):
        """
        Args:
            config: TrainerConfig
            split: Dataset split to use ("train" or "validation")
            collate_fn: Collate function to use for the dataset
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="right", use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.required_columns = ["input_ids", "attention_mask", "labels"]

        self.file_type = normalize_file_type(config.custom_dataset.file_type)
        self.data_path = (
            config.custom_dataset.train_dataset_path if split == "train" else config.custom_dataset.val_dataset_path
        )
        self.template = config.custom_dataset.template
        self.column_mapping = config.custom_dataset.column_mapping

        super().__init__(config, split, collate_fn)

    def _tokenize(self, example):
        prompt, output, full_text = build_prompt(
            example,
            template=self.template,
            column_mapping=self.column_mapping,
        )
        encoding = self.tokenizer(full_text, truncation=False, padding=False, return_tensors="pt")

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_encoding = self.tokenizer(prompt, truncation=False, padding=False, return_tensors="pt")
        prompt_input_ids = prompt_encoding["input_ids"].squeeze(0)
        prompt_len = prompt_input_ids.size(0)
        labels[:prompt_len] = -100

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["labels"] = labels
        example["len"] = input_ids.size(0)

        return example

    def _prepare_dataset(self):
        if self.data_path is None or self.data_path == "":
            if self.split == "train":
                raise ValueError("train_dataset_path is required and was not provided.")
            self.dataset = None
            return

        data_file = {self.split: self.data_path}
        raw_dataset = load_dataset(self.file_type, data_files=data_file, split=self.split)
        dataset_columns = set(raw_dataset[0].keys())

        self.column_mapping = resolve_column_mapping(self.template, self.column_mapping, dataset_columns)

        tokenized_dataset = raw_dataset.map(self._tokenize)
        filtered_dataset = tokenized_dataset.filter(lambda x: x["len"] <= self.config.max_length)
        filtered_dataset = filtered_dataset.remove_columns(
            [col for col in filtered_dataset.column_names if col not in self.required_columns]
        )
        if self.split == "train":
            filtered_dataset = filtered_dataset.shuffle(seed=self.config.seed)

        self.dataset = filtered_dataset

    def __len__(self):
        return len(self.dataset)

    def _get_dataloader(self):
        if self.dataset is None:
            return None

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, padding="max_length", max_length=self.config.max_length
        )

        if self.collate_fn is not None:
            total_collate_fn = lambda batch: self.collate_fn(data_collator(batch))
        else:
            total_collate_fn = data_collator

        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=total_collate_fn,
            shuffle=self.split == "train",
            drop_last=True,
        )
