# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from string import Template
from typing import Dict, overload

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig
from datasets import load_dataset

PROMPT_TEMPLATE = Template(
    """### Instruction:\n
Generate an SQL query for the given question and database schema.\n\n
### Input:\n
Question: $prompt\n
Schema: $context\n\n
### Output:\n"""
)
DATASET_PATH = "gretelai/synthetic_text_to_sql"


class TextToSQLDataset(BaseDataset):
    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        """
        Args:
            config: Training configuration
            split: Dataset split to use ("train", "test")
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, padding_side="right", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.required_columns = ["input_ids", "attention_mask", "labels"]
        self.split = split
        self.collate_fn = collate_fn

        self._prepare_dataset()

    def _tokenize_function(self, example: Dict) -> Dict:
        prompt = example["sql_prompt"]
        context = example.get("sql_context", "")
        sql = example["sql"]

        input_text = PROMPT_TEMPLATE.substitute(prompt=prompt, context=context)
        target_text = sql.strip()
        full_text = input_text + target_text

        encoding = self.tokenizer(full_text, truncation=False, padding=False, return_tensors="pt")

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        prompt_encoding = self.tokenizer(input_text, truncation=False, padding=False, return_tensors="pt")
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
        raw_dataset = load_dataset(DATASET_PATH, split=self.split)
        raw_dataset = raw_dataset.filter(lambda example: example["sql_complexity"] == "basic SQL")

        tokenized_dataset = raw_dataset.map(self._tokenize_function)
        self.full_dataset = tokenized_dataset.filter(lambda example: example["len"] <= self.config.max_length)
        self.dataset = self.full_dataset.remove_columns(
            [col for col in self.full_dataset.column_names if col not in self.required_columns]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]

        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["labels"],
        }

    @overload
    def _get_dataloader(self) -> DataLoader:
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
