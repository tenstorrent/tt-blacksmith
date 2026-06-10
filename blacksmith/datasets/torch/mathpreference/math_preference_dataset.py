# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from blacksmith.datasets.torch.mathpreference.math_preference_utils import (
    DATASET_PATH,
    IGNORED_LABEL_ID,
    PROMPT_TEMPLATE,
)
from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig
from datasets import Dataset, load_dataset


class DatasetMode(Enum):
    """Mode for the math preference dataset."""

    DPO = "dpo"  # Returns chosen + rejected for DPO training
    SFT = "sft"  # Returns only chosen for SFT training


TRAIN_VAL_SPLIT_RATIO = 0.9
DEFAULT_SFT_RATIO = 0.33


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

    # Cache raw source dataset filtered by both chosen/rejected lengths.
    _cached_filtered_source_dataset: Dataset = None
    _cached_filter_key: Tuple[str, int] = None

    # Cache fully tokenized datasets by mode + tokenizer constraints.
    _cached_full_dataset_by_mode: Dict[Tuple[DatasetMode, str, int], Dataset] = {}

    def __init__(
        self,
        config: TrainingConfig,
        split: str = "train",
        collate_fn=None,
        mode: str = "dpo",
        sft_ratio: float = DEFAULT_SFT_RATIO,
    ):
        """
        Args:
            config: Training configuration
            split: Dataset split to use ("train", "validation")
            collate_fn: Optional additional collate function
            mode: "dpo" for DPO training, "sft" for SFT training
            sft_ratio: Fraction in [0, 1] used to split filtered train data
                into SFT and DPO pools (default: 0.33).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="right", use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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

        if not (0.0 <= sft_ratio <= 1.0):
            raise ValueError(f"Invalid sft_ratio={sft_ratio}. Expected a value in range [0, 1].")
        self.sft_ratio = sft_ratio

        super().__init__(config, split, collate_fn)

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

    def _with_sequence_lengths(self, example: Dict) -> Dict:
        """Annotate raw sample with chosen/rejected sequence lengths."""
        instruction = example["instruction"]
        chosen = example["chosen_response"]
        rejected = example["rejected_response"]
        prompt = self._format_prompt(instruction)

        chosen_input_ids = self.tokenizer(prompt + chosen, truncation=False, padding=False)["input_ids"]
        rejected_input_ids = self.tokenizer(prompt + rejected, truncation=False, padding=False)["input_ids"]
        example["chosen_len"] = len(chosen_input_ids)
        example["rejected_len"] = len(rejected_input_ids)
        return example

    def _prepare_filtered_source_dataset(self) -> Dataset:
        """
        Prepare source train split filtered by BOTH chosen and rejected lengths.
        This filtered pool is used as the basis for SFT/DPO phase split.
        """
        filter_key = (self.config.model_name, self.config.max_length)
        if (
            MathPreferenceDataset._cached_filtered_source_dataset is not None
            and MathPreferenceDataset._cached_filter_key == filter_key
        ):
            return MathPreferenceDataset._cached_filtered_source_dataset

        raw_dataset = load_dataset(DATASET_PATH, split="train")
        dataset_with_lengths = raw_dataset.map(
            self._with_sequence_lengths, desc="Computing chosen/rejected lengths"
        )
        filtered_dataset = dataset_with_lengths.filter(
            lambda example: example["chosen_len"] <= self.config.max_length
            and example["rejected_len"] <= self.config.max_length
        )

        MathPreferenceDataset._cached_filtered_source_dataset = filtered_dataset
        MathPreferenceDataset._cached_filter_key = filter_key
        return filtered_dataset

    def _prepare_mode_dataset(self) -> Dataset:
        """Prepare and cache the full tokenized dataset for the selected mode."""
        cache_key = (self.mode, self.config.model_name, self.config.max_length)
        if cache_key in MathPreferenceDataset._cached_full_dataset_by_mode:
            return MathPreferenceDataset._cached_full_dataset_by_mode[cache_key]

        # Source dataset pool is filtered by both chosen and rejected lengths.
        raw_dataset = self._prepare_filtered_source_dataset()

        if self.mode == DatasetMode.DPO:
            # Tokenize for DPO (both chosen and rejected)
            tokenized_dataset = raw_dataset.map(self._tokenize_dpo, desc="Tokenizing preference pairs for DPO")

            full_dataset = tokenized_dataset
        else:
            # Tokenize for SFT (chosen only)
            tokenized_dataset = raw_dataset.map(self._tokenize_sft, desc="Tokenizing chosen responses for SFT")

            full_dataset = tokenized_dataset

        # Remove columns not needed for training
        full_dataset = full_dataset.remove_columns(
            [col for col in full_dataset.column_names if col not in self.required_columns]
        )
        full_dataset = full_dataset.shuffle(seed=self.config.seed)

        MathPreferenceDataset._cached_full_dataset_by_mode[cache_key] = full_dataset
        return full_dataset

    def _select_mode_phase_dataset(self, full_dataset: Dataset) -> Dataset:
        """Split full train data into SFT and DPO pools."""
        length = len(full_dataset)
        sft_boundary = int(length * self.sft_ratio)

        if self.mode == DatasetMode.SFT:
            return full_dataset.select(range(0, sft_boundary))
        return full_dataset.select(range(sft_boundary, length))

    def _select_split_dataset(self, phase_dataset: Dataset) -> Dataset:
        """Split a mode-specific pool into train/validation subsets."""
        if getattr(self.config, "disable_validation", False):
            if self.split == "train":
                return phase_dataset
            if self.split == "validation":
                return phase_dataset.select(range(0))

        length = len(phase_dataset)
        train_val_split = int(TRAIN_VAL_SPLIT_RATIO * length)

        if self.split == "train":
            return phase_dataset.select(range(0, train_val_split))
        if self.split == "validation":
            return phase_dataset.select(range(train_val_split, length))
        raise ValueError(
            f"Invalid split '{self.split}' for MathPreferenceDataset. Only 'train' and 'validation' are supported."
        )

    def _prepare_dataset(self):
        """Load and prepare dataset with SFT/DPO and train/validation splitting."""
        full_dataset = self._prepare_mode_dataset()
        phase_dataset = self._select_mode_phase_dataset(full_dataset)
        self.dataset = self._select_split_dataset(phase_dataset)

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

    def _get_dataloader(self) -> DataLoader:
        """Create DataLoader for training."""
        if self.mode == DatasetMode.DPO:
            data_collator = DPODataCollator(tokenizer=self.tokenizer, max_length=self.config.max_length)
        else:
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


# Backward compatibility aliases
MathDPODataset = lambda config, split="train", collate_fn=None, sft_ratio=DEFAULT_SFT_RATIO: MathPreferenceDataset(
    config, split, collate_fn, mode="dpo", sft_ratio=sft_ratio
)
MathSFTDataset = lambda config, split="train", collate_fn=None, sft_ratio=DEFAULT_SFT_RATIO: MathPreferenceDataset(
    config, split, collate_fn, mode="sft", sft_ratio=sft_ratio
)