# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Math Preference Dataset for DPO and SFT training.

Based on the paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/pdf/2305.18290

Dataset: argilla/distilabel-math-preference-dpo

Supports two modes:
- "dpo": Returns both chosen and rejected responses for DPO training
- "sft": Returns only chosen responses for SFT training (stage 1 of DPO pipeline)
"""
from enum import Enum
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.datasets.torch.dpo.math_dpo_utils import (
    DATASET_PATH,
    PROMPT_TEMPLATE,
    IGNORED_LABEL_ID,
)
from blacksmith.tools.templates.configs import TrainingConfig


class DatasetMode(Enum):
    """Mode for the math preference dataset."""

    DPO = "dpo"  # Returns chosen + rejected for DPO training
    SFT = "sft"  # Returns only chosen for SFT training


class DPODataCollator:
    """
    Custom data collator for DPO training.

    Handles padding for both chosen and rejected sequences.
    """

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def _pad_sequence(self, sequences: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        """Pad a list of sequences to max_length."""
        batch_size = len(sequences)
        padded = torch.full((batch_size, self.max_length), pad_value, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            length = min(len(seq), self.max_length)
            padded[i, :length] = seq[:length]
        return padded

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of DPO samples."""
        chosen_input_ids = [sample["chosen_input_ids"] for sample in batch]
        chosen_attention_mask = [sample["chosen_attention_mask"] for sample in batch]
        chosen_labels = [sample["chosen_labels"] for sample in batch]

        rejected_input_ids = [sample["rejected_input_ids"] for sample in batch]
        rejected_attention_mask = [sample["rejected_attention_mask"] for sample in batch]
        rejected_labels = [sample["rejected_labels"] for sample in batch]

        return {
            "chosen_input_ids": self._pad_sequence(chosen_input_ids, self.pad_token_id),
            "chosen_attention_mask": self._pad_sequence(chosen_attention_mask, 0),
            "chosen_labels": self._pad_sequence(chosen_labels, IGNORED_LABEL_ID),
            "rejected_input_ids": self._pad_sequence(rejected_input_ids, self.pad_token_id),
            "rejected_attention_mask": self._pad_sequence(rejected_attention_mask, 0),
            "rejected_labels": self._pad_sequence(rejected_labels, IGNORED_LABEL_ID),
        }


class MathPreferenceDataset(BaseDataset):
    """
    Dataset for math preference data supporting both DPO and SFT modes.

    Modes:
    - DPO mode: Returns chosen + rejected responses for preference learning
    - SFT mode: Returns only chosen responses for supervised fine-tuning

    Each sample in the source dataset contains:
    - instruction: The math problem/question
    - chosen_response: The preferred (better) response
    - rejected_response: The less preferred response
    """

    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None, mode: str = "dpo"):
        """
        Args:
            config: Training configuration
            split: Dataset split to use ("train")
            collate_fn: Optional additional collate function
            mode: "dpo" for DPO training, "sft" for SFT training
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, padding_side="right", use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.split = split
        self.collate_fn = collate_fn
        self.max_length = config.max_length

        # Set mode
        if isinstance(mode, str):
            self.mode = DatasetMode(mode.lower())
        else:
            self.mode = mode

        # Set required columns based on mode
        if self.mode == DatasetMode.DPO:
            self.required_columns = [
                "chosen_input_ids",
                "chosen_attention_mask",
                "chosen_labels",
                "rejected_input_ids",
                "rejected_attention_mask",
                "rejected_labels",
            ]
        else:  # SFT mode
            self.required_columns = ["input_ids", "attention_mask", "labels"]

        self._prepare_dataset()

    def _format_prompt(self, instruction: str) -> str:
        """Format the instruction as a prompt."""
        return PROMPT_TEMPLATE.substitute(instruction=instruction)

    def _tokenize_dpo(self, example: Dict) -> Dict:
        """
        Tokenize a preference pair for DPO training.

        Returns both chosen and rejected sequences with labels.
        """
        instruction = example["instruction"]
        chosen = example["chosen_response"]
        rejected = example["rejected_response"]

        prompt = self._format_prompt(instruction)

        # Tokenize prompt to get its length
        prompt_encoding = self.tokenizer(prompt, truncation=False, padding=False, return_tensors="pt")
        prompt_len = prompt_encoding["input_ids"].size(1)

        # Tokenize chosen response
        chosen_full = prompt + chosen
        chosen_encoding = self.tokenizer(chosen_full, truncation=False, padding=False, return_tensors="pt")
        chosen_input_ids = chosen_encoding["input_ids"].squeeze(0)
        chosen_attention_mask = chosen_encoding["attention_mask"].squeeze(0)

        # Chosen labels (mask the prompt part)
        chosen_labels = chosen_input_ids.clone()
        chosen_labels[:prompt_len] = IGNORED_LABEL_ID

        # Tokenize rejected response
        rejected_full = prompt + rejected
        rejected_encoding = self.tokenizer(rejected_full, truncation=False, padding=False, return_tensors="pt")
        rejected_input_ids = rejected_encoding["input_ids"].squeeze(0)
        rejected_attention_mask = rejected_encoding["attention_mask"].squeeze(0)

        # Rejected labels (mask the prompt part)
        rejected_labels = rejected_input_ids.clone()
        rejected_labels[:prompt_len] = IGNORED_LABEL_ID

        example["chosen_input_ids"] = chosen_input_ids
        example["chosen_attention_mask"] = chosen_attention_mask
        example["chosen_labels"] = chosen_labels
        example["rejected_input_ids"] = rejected_input_ids
        example["rejected_attention_mask"] = rejected_attention_mask
        example["rejected_labels"] = rejected_labels
        example["chosen_len"] = chosen_input_ids.size(0)
        example["rejected_len"] = rejected_input_ids.size(0)

        return example

    def _tokenize_sft(self, example: Dict) -> Dict:
        """
        Tokenize only the chosen response for SFT training.

        Returns standard SFT format (input_ids, attention_mask, labels).
        """
        instruction = example["instruction"]
        chosen = example["chosen_response"]

        prompt = self._format_prompt(instruction)
        full_text = prompt + chosen

        # Tokenize full text
        encoding = self.tokenizer(full_text, truncation=False, padding=False, return_tensors="pt")
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Tokenize prompt to get its length
        prompt_encoding = self.tokenizer(prompt, truncation=False, padding=False, return_tensors="pt")
        prompt_len = prompt_encoding["input_ids"].size(1)

        # Create labels (mask the prompt part with -100)
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORED_LABEL_ID

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["labels"] = labels
        example["len"] = input_ids.size(0)

        return example

    def _prepare_dataset(self):
        """Load and prepare the dataset based on mode."""
        raw_dataset = load_dataset(DATASET_PATH, split=self.split)

        if self.mode == DatasetMode.DPO:
            # Tokenize for DPO (both chosen and rejected)
            tokenized_dataset = raw_dataset.map(self._tokenize_dpo, desc="Tokenizing preference pairs for DPO")

            # Filter samples that exceed max_length
            self.full_dataset = tokenized_dataset.filter(
                lambda example: example["chosen_len"] <= self.max_length and example["rejected_len"] <= self.max_length
            )
        else:
            # Tokenize for SFT (chosen only)
            tokenized_dataset = raw_dataset.map(self._tokenize_sft, desc="Tokenizing chosen responses for SFT")

            # Filter samples that exceed max_length
            self.full_dataset = tokenized_dataset.filter(lambda example: example["len"] <= self.max_length)

        # Remove columns not needed for training
        self.dataset = self.full_dataset.remove_columns(
            [col for col in self.full_dataset.column_names if col not in self.required_columns]
        )

        # Set format for PyTorch
        self.dataset.set_format(type="torch")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]
        if self.mode == DatasetMode.SFT:
            return {
                "input_ids": sample["input_ids"],
                "attention_mask": sample["attention_mask"],
                "labels": sample["labels"],
            }
        return sample

    def get_dataloader(self) -> DataLoader:
        """Create DataLoader for training."""
        if self.mode == DatasetMode.DPO:
            data_collator = DPODataCollator(tokenizer=self.tokenizer, max_length=self.max_length)
        else:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer, padding="max_length", max_length=self.max_length
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


# Backward compatibility aliases
MathDPODataset = lambda config, split="train", collate_fn=None: MathPreferenceDataset(
    config, split, collate_fn, mode="dpo"
)
MathSFTDataset = lambda config, split="train", collate_fn=None: MathPreferenceDataset(
    config, split, collate_fn, mode="sft"
)
