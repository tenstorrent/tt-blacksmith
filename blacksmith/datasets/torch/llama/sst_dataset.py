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
        """Apply prompt template to dataset examples."""

        prompt = PROMPT_TEMPLATE.substitute(input=example["sentence"])
        response = RESPONSE_TEMPLATE.substitute(label=LBL2VALUE[example["label"]])
        example["text"] = prompt + response
        example["prompt"] = prompt

        return example

    def _tokenize_function(self, examples: Dict[str, Any], mode: str = "train") -> Dict[str, Any]:
        """Tokenize input and create labels with masked prompt tokens."""

        tokenized_batch = self.tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=self.config.max_length
        )

        prompt_encodings = self.tokenizer(
            examples["prompt"], padding="max_length", truncation=True, max_length=self.config.max_length
        )

        labels = []
        for input_ids, prompt_ids in zip(tokenized_batch["input_ids"], prompt_encodings["input_ids"]):
            label = input_ids.copy()
            for idx, prompt_id in enumerate(prompt_ids):
                if prompt_id != self.tokenizer.pad_token_id:
                    label[idx] = -100  # mask prompt
                else:
                    break  # no need to mask padding or response
            # Also mask padding in the label
            label = [l if l != self.tokenizer.pad_token_id else -100 for l in label]
            labels.append(label)

        tokenized_batch["labels"] = labels

        return tokenized_batch

    def _filter_by_token_length(self, example: Dict[str, Any], max_tokens: int = 58) -> bool:
        """Filter examples by token length. Returns True if example should be kept."""
        
        # Tokenize the text to get token count
        tokens = self.tokenizer(example["text"], add_special_tokens=True)
        token_count = len(tokens["input_ids"])
        
        # Return True if within limit, False if too long
        return token_count <= max_tokens

    def load_tokenized_data(self) -> Tuple[Any, Any]:
        print(f"Loading dataset ({self.config.dataset_id})...")
        dataset = load_dataset(self.config.dataset_id)

        train_set = dataset["train"].map(self._apply_template)
        train_set = train_set.filter(self._filter_by_token_length)
        tokenized_train_set = train_set.map(self._tokenize_function, batched=True)
        tokenized_train_set.set_format("torch", columns=self.required_columns)

        # breakpoint()
        # self.tokenizer.decode(tokenized_train_set[0]["input_ids"])
        # self.tokenizer.decode([token if token != -100 else 0  for token in tokenized_train_set[0]["labels"]])

        validation_set = dataset["validation"].map(self._apply_template)
        validation_set = validation_set.filter(self._filter_by_token_length)
        tokenized_validation_set = validation_set.map(self._tokenize_function, batched=True)
        tokenized_validation_set.set_format("torch", columns=self.required_columns)

        return tokenized_train_set, tokenized_validation_set

#     def load_test_data(self):
#         print(f"Loading test dataset ({self.config.dataset_id})...")
#         dataset = load_dataset(self.config.dataset_id, split="test")
# 
#         test_set = dataset.map(self._apply_template)
#         test_set = test_set.filter(self._filter_by_token_length)
#         tokenized_test_set = test_set.map(self._tokenize_function, batched=True)
#         tokenized_test_set.set_format("torch", columns=self.required_columns)
# 
#         return tokenized_test_set