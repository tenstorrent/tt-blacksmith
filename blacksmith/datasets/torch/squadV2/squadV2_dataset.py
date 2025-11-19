# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from string import Template
from typing import Dict

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from blacksmith.tools.templates.configs import TrainingConfig
from blacksmith.datasets.torch.torch_dataset import BaseDataset


PROMPT_TEMPLATE = Template(
    """
Context: $context\n
Question: $question\n
Answer:
"""
)


class SquadV2Dataset(BaseDataset):
    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        """
        Args:
            config: TrainingConfig (ensure config.dataset_id is set to "squad_v2")
            split: Dataset split to use ("train", "validation")
            collate_fn: Collate function to use for the dataset
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, padding_side="right", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.required_columns = ["input_ids", "attention_mask", "labels"]
        self.split = split
        self.collate_fn = collate_fn

        self._prepare_dataset()

    def _tokenize_function(self, example):
        context = example["context"]
        question = example["question"]
        prompt = PROMPT_TEMPLATE.substitute(context=context, question=question)

        # Determine the response
        # SQuAD v2.0 has unanswerable questions, indicated by an empty 'text' list.
        if example["answers"]["text"]:
            response = example["answers"]["text"][0]
        else:
            response = "unanswerable"

        full_text = prompt + response

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
        example["full_text"] = full_text
        example["len"] = input_ids.size(0)

        return example

    def _prepare_dataset(self):
        raw_dataset = load_dataset(self.config.dataset_id, split=self.split)

        # reduce the size of the validation dataset
        if self.split == "validation":
            raw_dataset = raw_dataset.train_test_split(test_size=0.02, seed=self.config.seed)["test"]

        tokenized_dataset = raw_dataset.map(self._tokenize_function)
        self.full_dataset = tokenized_dataset.filter(lambda example: example["len"] <= self.config.max_length)
        self.dataset = self.full_dataset.remove_columns(
            [col for col in self.full_dataset.column_names if col not in self.required_columns]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["labels"],
        }

    def get_dataloader(self) -> DataLoader:
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
