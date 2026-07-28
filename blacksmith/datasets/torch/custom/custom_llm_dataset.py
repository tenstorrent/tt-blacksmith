# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from blacksmith.datasets.torch.custom.custom_dataset_utils import build_alpaca_prompt
from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig
from datasets import load_dataset


class CustomLLMDataset(BaseDataset):
    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        """
        Args:
            config: TrainingConfig
            split: Dataset split to use ("train" or "validation")
            collate_fn: Collate function to use for the dataset
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="right", use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.required_columns = ["input_ids", "attention_mask", "labels"]

        self.file_type = config.dataset.file_type
        self.dataset_path = config.dataset.dataset_path
        self.column_mapping = config.dataset.column_mapping

        super().__init__(config, split, collate_fn)

    def _format_example(self, example):
        # TODO(tstepanovicTT): Make it work for any prompt format, not just Alpaca.
        instruction = example[self.column_mapping["instruction"]]
        input_col = self.column_mapping.get("input", "")
        input_text = example[input_col] if input_col else ""
        output = example[self.column_mapping["output"]]

        prompt = build_alpaca_prompt(instruction, input_text)
        full_text = prompt + output

        return prompt, output, full_text

    def _tokenize(self, example):
        prompt, _, full_text = self._format_example(example)
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
        raw_dataset = load_dataset(self.file_type, data_files=self.dataset_path, split="train")
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
            drop_last=self.config.dataset.drop_last,
        )
