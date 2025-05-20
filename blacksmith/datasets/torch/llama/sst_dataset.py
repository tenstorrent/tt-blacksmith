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

    def _apply_template(self, example: Dict[str, Any], mode: str = "train") -> Dict[str, Any]:
        """Apply prompt template to dataset examples."""

        prompt = PROMPT_TEMPLATE.substitute(input=example["sentence"])
        if mode == "train":
            response = RESPONSE_TEMPLATE.substitute(label=LBL2VALUE[example["label"]])
            example["text"] = prompt + response
            example["prompt"] = prompt
        else:
            example["text"] = prompt

        return example

    def _tokenize_function(self, example: Dict[str, Any], mode: str = "train") -> Dict[str, Any]:
        """Tokenize input and create labels with masked prompt tokens."""

        tokenized_batch = self.tokenizer(
            example["text"], padding="max_length", truncation=True, max_length=self.config.max_length
        )

        if mode == "test":
            example["label"] = [LBL2VALUE[lbl] for lbl in example["label"]]
            tokenized_lbls = self.tokenizer(
                example["label"], padding="max_length", max_length=self.config.max_length, truncation=True
            )
            tokenized_batch["labels"] = tokenized_lbls["input_ids"]

            return tokenized_batch

        prompt_encodings = self.tokenizer(
            example["prompt"], padding="max_length", truncation=True, max_length=self.config.max_length
        )

        input_ids = torch.tensor(tokenized_batch["input_ids"])
        prompt_ids = torch.tensor(prompt_encodings["input_ids"])

        labels = input_ids.clone()
        prompt_len = (prompt_ids[0] != self.tokenizer.pad_token_id).sum()
        mask = torch.arange(input_ids.size(1)) < prompt_len
        labels[:, mask] = -100
        tokenized_batch["labels"] = list(labels.unbind(0))

        return tokenized_batch

    def load_tokenized_data(self) -> Tuple[Any, Any]:
        print(f"Loading dataset ({self.config.dataset_id})...")
        dataset = load_dataset(self.config.dataset_id)

        # Labels are just expected values (0 or 1)
        train_set = dataset["train"].map(self._apply_template, fn_kwargs={"mode": "train"})
        tokenized_train_set = train_set.map(self._tokenize_function, fn_kwargs={"mode": "train"}, batched=True)
        tokenized_train_set.set_format("torch", columns=self.required_columns)

        eval_set = dataset["validation"].map(self._apply_template, fn_kwargs={"mode": "test"})

        return tokenized_train_set, eval_set
