# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig
from transformers import AutoTokenizer
from datasets import load_dataset
from blacksmith.datasets.torch.fusechat.fusechat_utils import DATASET_PATH, PROMPT_TEMPLATE
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq

class FuseChatDataset(BaseDataset):
    # FuseChat dataset only has train split, so we create validation/test from it.
    # This is used to avoid reloading the dataset multiple times.
    _shared_dataset = None

    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        """
        Args:
            config: TrainingConfig (ensure config.dataset_id is set to "fusechat")
            split: Dataset split to use ("train")
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
        input = example[0]["value"]
        output = example[1]["output"]

        prompt = PROMPT_TEMPLATE.substitute(input=input)

        response = output
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
        # FuseChat only has train split, so we create validation/test from it.
        if FuseChatDataset._shared_dataset is None:
            raw_dataset = load_dataset(DATASET_PATH, split="train")
            tokenized_dataset = raw_dataset.map(self._tokenize_function)
            filtered_dataset = tokenized_dataset.filter(lambda x: x["len"] <= self.config.max_length)
            filtered_dataset = filtered_dataset.shuffle(seed=self.config.seed)
            filtered_dataset = filtered_dataset.remove_columns(
                [col for col in filtered_dataset.column_names if col not in self.required_columns]
            )
            FuseChatDataset._shared_dataset = filtered_dataset
        
        full_dataset = FuseChatDataset._shared_dataset
        n = len(full_dataset)
        train_end = int(0.98 * n)
        val_end = n
        if self.split == "train":
            self.dataset = full_dataset.select(range(0, train_end))
        elif self.split == "validation":
            self.dataset = full_dataset.select(range(train_end, val_end))
        else:
            raise ValueError(f"Invalid split '{self.split}' for AlpacaDataset. Only 'train' and 'validation' are supported.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["labels"],
        }

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
