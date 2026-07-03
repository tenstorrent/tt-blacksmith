# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GSM8K dataset for GRPO reasoning training.

Unlike the supervised datasets, GRPO generates its own completions at train
time, so this dataset is PROMPT-ONLY: each item is a chat-templated, R1-style
prompt plus the gold integer answer used by the correctness reward. Prompts are
left-padded at collate time so batched autoregressive generation shares a single
generation frontier.

Dataset: https://huggingface.co/datasets/openai/gsm8k
"""
from typing import Dict, List

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from blacksmith.datasets.torch.gsm8k.gsm8k_utils import (
    DATASET_CONFIG,
    DATASET_PATH,
    GSM8K_SYSTEM_PROMPT,
    extract_gsm8k_gold,
)
from blacksmith.datasets.torch.torch_dataset import BaseDataset
from datasets import load_dataset


class GSM8KDataset(BaseDataset):
    """Prompt-only GSM8K dataset yielding ``{"prompt", "gold"}`` items."""

    def __init__(self, config, split: str = "train", collate_fn=None):
        # Left padding is required for batched generation: it aligns the last real
        # prompt token to a common index across the batch.
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="left", use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        super().__init__(config, split, collate_fn)

    def _build_prompt(self, question: str) -> str:
        # Gemma chat templates do not support a system role, so the R1 instructions
        # are folded into the user turn.
        messages = [{"role": "user", "content": f"{GSM8K_SYSTEM_PROMPT}\n\nQuestion: {question}"}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _prepare_dataset(self):
        # GSM8K ships only train/test splits.
        hf_split = "train" if self.split == "train" else "test"
        raw = load_dataset(DATASET_PATH, DATASET_CONFIG, split=hf_split)

        def _map(example):
            return {
                "prompt": self._build_prompt(example["question"]),
                "gold": extract_gsm8k_gold(example["answer"]),
            }

        self.dataset = raw.map(_map, remove_columns=raw.column_names, desc="Building GSM8K prompts")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx]

    def _collate(self, batch: List[Dict]) -> Dict:
        prompts = [item["prompt"] for item in batch]
        golds = [item["gold"] for item in batch]
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_prompt_length,
            return_tensors="pt",
            add_special_tokens=False,  # apply_chat_template already added them
        )
        return {
            "prompt_input_ids": encoded["input_ids"],
            "prompt_attention_mask": encoded["attention_mask"],
            "gold_answers": golds,
        }

    def _get_dataloader(self) -> DataLoader:
        if self.collate_fn is not None:
            collate = lambda b: self.collate_fn(self._collate(b))
        else:
            collate = self._collate

        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate,
            shuffle=self.split == "train",
            drop_last=True,
        )
