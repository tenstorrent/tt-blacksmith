# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from blacksmith.datasets.torch.mathpreference.math_preference_utils import (
    DATASET_PATH,
    DEFAULT_SFT_RATIO,
    DPO_REQUIRED_COLUMNS,
    IGNORED_LABEL_ID,
    PROMPT_TEMPLATE,
    SFT_REQUIRED_COLUMNS,
    TRAIN_VAL_SPLIT_RATIO,
    DatasetMode,
)
from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig
from datasets import Dataset, load_dataset


class DPODataCollator:
    """
    Data collator for DPO training.

    Pads the chosen and rejected sequences independently by delegating to
    ``DataCollatorForSeq2Seq`` (the same collator used for SFT), then merges the
    two padded groups back into the ``chosen_*`` / ``rejected_*`` layout.
    """

    def __init__(self, tokenizer, max_length: int):
        self.collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="max_length",
            max_length=max_length,
            label_pad_token_id=IGNORED_LABEL_ID,
        )

    def _collate_side(self, batch: List[Dict], prefix: str) -> Dict[str, torch.Tensor]:
        """Pad a single side (chosen or rejected) using the canonical seq2seq keys."""
        features = [
            {
                "input_ids": sample[f"{prefix}_input_ids"],
                "attention_mask": sample[f"{prefix}_attention_mask"],
                "labels": sample[f"{prefix}_labels"],
            }
            for sample in batch
        ]
        padded = self.collator(features)
        return {
            f"{prefix}_input_ids": padded["input_ids"],
            f"{prefix}_attention_mask": padded["attention_mask"],
            f"{prefix}_labels": padded["labels"],
        }

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of DPO samples."""
        return {
            **self._collate_side(batch, "chosen"),
            **self._collate_side(batch, "rejected"),
        }


class MathPreferenceDataset(BaseDataset):
    """
    Dataset for Math Preference data supporting both DPO and SFT modes.

    Modes:
    - DPO mode: Returns chosen + rejected responses for DPO training.
    - SFT mode: Returns only chosen responses for SFT training.
    """

    # Cache the tokenized, mode-selected pool for the current mode, shared
    # across the train/validation instances to avoid re-tokenizing per run.
    _shared_dataset: Dataset = None

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
                into SFT and DPO pools
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="right", use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Set mode.
        if isinstance(mode, str):
            self.mode = DatasetMode(mode.lower())
        else:
            self.mode = mode

        # Set required columns based on mode.
        if self.mode == DatasetMode.DPO:
            self.required_columns = list(DPO_REQUIRED_COLUMNS)
        else:  # SFT mode.
            self.required_columns = list(SFT_REQUIRED_COLUMNS)

        if not (0.0 <= sft_ratio <= 1.0):
            raise ValueError(f"Invalid sft_ratio={sft_ratio}. Expected a value in range [0, 1].")
        self.sft_ratio = sft_ratio

        super().__init__(config, split, collate_fn)

    def _format_prompt(self, instruction: str) -> str:
        """Format the instruction as a prompt."""
        return PROMPT_TEMPLATE.substitute(instruction=instruction)

    def _tokenize(self, example: Dict) -> Dict:
        """
        Tokenize a preference pair.

        Returns both chosen and rejected sequences with labels. The SFT path
        later derives its inputs from the chosen side.
        """
        instruction = example["instruction"]
        chosen = example["chosen_response"]
        rejected = example["rejected_response"]

        prompt = self._format_prompt(instruction)

        # Tokenize prompt to get its length.
        prompt_encoding = self.tokenizer(prompt, truncation=False, padding=False, return_tensors="pt")
        prompt_len = prompt_encoding["input_ids"].size(1)

        # Tokenize chosen response.
        chosen_full = prompt + chosen
        chosen_encoding = self.tokenizer(chosen_full, truncation=False, padding=False, return_tensors="pt")
        chosen_input_ids = chosen_encoding["input_ids"].squeeze(0)
        chosen_attention_mask = chosen_encoding["attention_mask"].squeeze(0)

        # Chosen labels (mask the prompt part).
        chosen_labels = chosen_input_ids.clone()
        chosen_labels[:prompt_len] = IGNORED_LABEL_ID

        # Tokenize rejected response.
        rejected_full = prompt + rejected
        rejected_encoding = self.tokenizer(rejected_full, truncation=False, padding=False, return_tensors="pt")
        rejected_input_ids = rejected_encoding["input_ids"].squeeze(0)
        rejected_attention_mask = rejected_encoding["attention_mask"].squeeze(0)

        # Rejected labels (mask the prompt part).
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

    def _select_split_dataset(self, phase_dataset: Dataset) -> Dataset:
        """Split a mode-specific pool into train/validation subsets."""
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
        """Load and prepare dataset with SFT/DPO and train/validation splitting.

        Tokenization always produces both the chosen and rejected sequences (the
        SFT path derives its inputs from the chosen side afterwards). Rows whose
        chosen or rejected sequence exceeds ``max_length`` are filtered out using
        the lengths produced during tokenization, so the text is tokenized only
        once. The shuffled pool is then split into the SFT/DPO phase for the
        current mode and cached; a single script only runs one mode, so the
        train and validation instances share this phase-selected pool.
        """
        if MathPreferenceDataset._shared_dataset is None:
            raw_dataset = load_dataset(DATASET_PATH, split="train")
            tokenized_dataset = raw_dataset.map(self._tokenize, desc="Tokenizing preference pairs")

            # Drop pairs that do not fit within max_length (using the lengths the
            # tokenization step already computed).
            tokenized_dataset = tokenized_dataset.filter(
                lambda example: example["chosen_len"] <= self.config.max_length
                and example["rejected_len"] <= self.config.max_length
            )

            # Keep only the tensors needed for training (drops the length columns).
            tokenized_dataset = tokenized_dataset.remove_columns(
                [col for col in tokenized_dataset.column_names if col not in DPO_REQUIRED_COLUMNS]
            )
            tokenized_dataset = tokenized_dataset.shuffle(seed=self.config.seed)

            # Split into the SFT/DPO phase pools and keep only this mode's pool.
            sft_boundary = int(len(tokenized_dataset) * self.sft_ratio)
            if self.mode == DatasetMode.SFT:
                tokenized_dataset = tokenized_dataset.select(range(0, sft_boundary))
            else:
                tokenized_dataset = tokenized_dataset.select(range(sft_boundary, len(tokenized_dataset)))

            MathPreferenceDataset._shared_dataset = tokenized_dataset

        split_dataset = self._select_split_dataset(MathPreferenceDataset._shared_dataset)

        # SFT keeps only the chosen side, renamed to the canonical column names
        # by stripping the "chosen_" prefix (e.g. chosen_input_ids -> input_ids).
        if self.mode == DatasetMode.SFT:
            chosen_columns = [col for col in split_dataset.column_names if col.startswith("chosen_")]
            split_dataset = split_dataset.remove_columns(
                [col for col in split_dataset.column_names if col not in chosen_columns]
            ).rename_columns({col: col[len("chosen_") :] for col in chosen_columns})

        self.dataset = split_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx]

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


# Backward compatibility aliases.
MathDPODataset = lambda config, split="train", collate_fn=None, sft_ratio=DEFAULT_SFT_RATIO: MathPreferenceDataset(
    config, split, collate_fn, mode="dpo", sft_ratio=sft_ratio
)
MathSFTDataset = lambda config, split="train", collate_fn=None, sft_ratio=DEFAULT_SFT_RATIO: MathPreferenceDataset(
    config, split, collate_fn, mode="sft", sft_ratio=sft_ratio
)
