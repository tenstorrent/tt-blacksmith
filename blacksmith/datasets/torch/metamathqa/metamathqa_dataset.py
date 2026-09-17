# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from inspect import cleandoc
from string import Template

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig
from datasets import Dataset, load_dataset

PROMPT_TEMPLATE = Template(
    cleandoc(
        """
        Below is an instruction that describes a task.
        Write a response that appropriately completes the request.

        ### Instruction:
        $instruction

        ### Response:
        $response
        """
    )
)

DATASET_PATH = "meta-math/MetaMathQA"

TRAIN_VAL_SPLIT_RATIO = 0.98


class MetaMathQADataset(BaseDataset):
    # MetaMathQA dataset only has train split, so we create validation/test from it.
    # This is used to avoid reloading the dataset multiple times.
    _cached_full_dataset: Dataset = None

    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        """
        Args:
            config: TrainingConfig (ensure config.dataset_id is set to "metamathqa")
            split: Dataset split to use ("train", "validation")
            collate_fn: Collate function to use for the dataset
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="right", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.required_columns = ["input_ids", "attention_mask", "labels"]

        super().__init__(config, split, collate_fn)

    def _tokenize_function(self, example):
        prompt = PROMPT_TEMPLATE.substitute(instruction=example["query"], response="")
        full_text = PROMPT_TEMPLATE.substitute(instruction=example["query"], response=example["response"])

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
        if MetaMathQADataset._cached_full_dataset is None:
            raw_dataset = load_dataset(DATASET_PATH, split="train")
            tokenized_dataset = raw_dataset.map(self._tokenize_function)
            filtered_dataset = tokenized_dataset.filter(lambda example: example["len"] <= self.config.max_length)
            filtered_dataset = filtered_dataset.remove_columns(
                [col for col in filtered_dataset.column_names if col not in self.required_columns]
            )
            filtered_dataset = filtered_dataset.shuffle(seed=self.config.seed)
            MetaMathQADataset._cached_full_dataset = filtered_dataset

        full_dataset = MetaMathQADataset._cached_full_dataset
        length = len(full_dataset)
        train_val_split = int(TRAIN_VAL_SPLIT_RATIO * length)

        if self.split == "train":
            self.dataset = full_dataset.select(range(0, train_val_split))
        elif self.split == "validation":
            self.dataset = full_dataset.select(range(train_val_split, length))
        else:
            raise ValueError(
                f"Invalid split '{self.split}' for MetaMathQADataset. Only 'train' and 'validation' are supported."
            )

    def __len__(self):
        return len(self.dataset)

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
