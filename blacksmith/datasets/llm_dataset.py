# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, Optional, Tuple

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from blacksmith.configs import TrainerConfig
from blacksmith.datasets.torch_dataset import BaseDataset


class LLMDataset(BaseDataset):
    """Prompt/response dataset for causal-LM fine-tuning.

    Subclasses provide the raw HF dataset (`_load_raw_dataset`) and how one example
    maps to a prompt and a response (`_prompt_and_response`). Everything else is
    shared: tokenization with the prompt masked out of the labels and an EOS token
    appended, the `max_length` filter, the train/validation carve-out for sources
    that ship a single split, and the padded dataloader.
    """

    required_columns = ["input_ids", "attention_mask", "labels"]

    # Sources with only a train split carve validation out of it with this ratio.
    # None means the source has the requested split natively.
    train_val_split_ratio: Optional[float] = None
    # Per-subclass cache of the tokenized single-split source, so it is loaded once
    # for both splits.
    _shared_dataset: Optional[Dataset] = None

    def __init__(self, config: TrainerConfig, split: str = "train", collate_fn=None):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="right", use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        super().__init__(config, split, collate_fn)

    def _load_raw_dataset(self) -> Optional[Dataset]:
        """Return the raw HF dataset for `self.split`, or the whole source when
        `train_val_split_ratio` is set. None means the split is not configured."""
        raise NotImplementedError

    def _prompt_and_response(self, example: Dict) -> Tuple[str, str]:
        raise NotImplementedError

    def _render(self, example: Dict) -> Tuple[str, str]:
        """Return (prompt, text) where text is the full training string before EOS.
        Override when the two are not a plain concatenation (e.g. chat templates)."""
        prompt, response = self._prompt_and_response(example)
        return prompt, prompt + response

    def _tokenize_function(self, example: Dict) -> Dict:
        prompt, text = self._render(example)
        full_text = text + self.tokenizer.eos_token

        encoding = self.tokenizer(full_text, truncation=False, padding=False, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        prompt_encoding = self.tokenizer(prompt, truncation=False, padding=False, return_tensors="pt")
        prompt_len = prompt_encoding["input_ids"].squeeze(0).size(0)
        labels[:prompt_len] = -100

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["labels"] = labels
        example["full_text"] = full_text
        example["len"] = input_ids.size(0)
        return example

    def _tokenize_and_filter(self, raw_dataset: Dataset) -> Dataset:
        tokenized_dataset = raw_dataset.map(self._tokenize_function)
        filtered_dataset = tokenized_dataset.filter(lambda example: example["len"] <= self.config.max_length)
        return filtered_dataset.remove_columns(
            [col for col in filtered_dataset.column_names if col not in self.required_columns]
        )

    def _prepare_dataset(self):
        if self.train_val_split_ratio is None:
            raw_dataset = self._load_raw_dataset()
            self.dataset = None if raw_dataset is None else self._tokenize_and_filter(raw_dataset)
            return

        cls = type(self)
        if cls.__dict__.get("_shared_dataset") is None:
            cls._shared_dataset = self._tokenize_and_filter(self._load_raw_dataset()).shuffle(seed=self.config.seed)
        full_dataset = cls._shared_dataset
        train_end = int(self.train_val_split_ratio * len(full_dataset))
        if self.split == "train":
            self.dataset = full_dataset.select(range(0, train_end))
        elif self.split == "validation":
            self.dataset = full_dataset.select(range(train_end, len(full_dataset)))
        else:
            raise ValueError(
                f"Invalid split '{self.split}' for {cls.__name__}. Only 'train' and 'validation' are supported."
            )

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["labels"],
        }

    def _get_dataloader(self) -> Optional[DataLoader]:
        if self.dataset is None:
            return None

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, padding="max_length", max_length=self.config.max_length
        )
        if self.collate_fn is not None:
            total_collate_fn = lambda batch: self.collate_fn(data_collator(batch))  # noqa: E731
        else:
            total_collate_fn = data_collator

        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=total_collate_fn,
            shuffle=self.split == "train",
            drop_last=True,
        )
