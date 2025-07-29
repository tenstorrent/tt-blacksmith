# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

from blacksmith.datasets.torch.llama.sst_utils import PROMPT_TEMPLATE, RESPONSE_TEMPLATE, LBL2VALUE
from blacksmith.experiments.torch.llama.configs import TrainingConfig


class SSTDataset:
    def __init__(self, config: TrainingConfig):
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, padding_side="right", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.required_columns = ["input_ids", "attention_mask", "labels"]

    def _apply_template(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = PROMPT_TEMPLATE.substitute(input=example["sentence"])
        response = RESPONSE_TEMPLATE.substitute(label=LBL2VALUE[example["label"]])
        example["text"] = prompt + response
        example["prompt"] = prompt

        return example

    def _tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        tokenized_batch = self.tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=self.config.max_length
        )

        prompt_encodings = self.tokenizer(
            examples["prompt"], padding="max_length", truncation=True, max_length=self.config.max_length
        )

        # Create labels by masking prompt tokens
        labels = []
        for input_ids, prompt_ids in zip(tokenized_batch["input_ids"], prompt_encodings["input_ids"]):
            label = input_ids.copy()
            for idx, prompt_id in enumerate(prompt_ids):
                if prompt_id != self.tokenizer.pad_token_id:
                    label[idx] = -100
                else:
                    break

            # Also mask padding in the label
            label = [l if l != self.tokenizer.pad_token_id else -100 for l in label]
            labels.append(label)

        tokenized_batch["labels"] = labels

        return tokenized_batch

    def load_tokenized_data(self) -> Tuple[Any, Any]:
        print(f"Loading dataset ({self.config.dataset_id})...")
        dataset = load_dataset(self.config.dataset_id)

        train_set = dataset["train"].map(self._apply_template)
        tokenized_train_set = train_set.map(self._tokenize_function, batched=True)
        tokenized_train_set.set_format("torch", columns=self.required_columns)

        validation_set = dataset["validation"].map(self._apply_template)
        tokenized_validation_set = validation_set.map(self._tokenize_function, batched=True)
        tokenized_validation_set.set_format("torch", columns=self.required_columns)

        return tokenized_train_set, tokenized_validation_set
