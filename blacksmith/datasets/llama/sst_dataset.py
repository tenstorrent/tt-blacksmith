# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer

from blacksmith.datasets.llama.sst_utils import TRAIN_PROMPT_TEMPLATE, TEST_PROMPT_TEMPLATE, LBL2VALUE
from blacksmith.experiments.pytorch.llama.configs import TrainingConfig


class SSTDataset:
    def __init__(self, config: TrainingConfig):
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, padding_side="right", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.required_columns = ["input_ids", "attention_mask", "labels", "text", "label"]

    def _apply_template(self, example: Dict[str, Any], mode: str = "train") -> Dict[str, Any]:
        """Apply prompt template to dataset examples."""
        if mode == "train":
            example["text"] = TRAIN_PROMPT_TEMPLATE.substitute(
                input=example["sentence"], label=LBL2VALUE[example["label"]]
            )
        else:
            example["text"] = TEST_PROMPT_TEMPLATE.substitute(input=example["sentence"])
        return example

    def _tokenize_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize input text and labels."""
        tokenized_batch = self.tokenizer(
            example["text"], padding="max_length", max_length=self.config.max_length, truncation=True
        )

        # Convert labels to strings
        example["label"] = [LBL2VALUE[lbl] for lbl in example["label"]]
        tokenized_lbls = self.tokenizer(
            example["label"], padding="max_length", max_length=self.config.max_length, truncation=True
        )

        tokenized_batch["labels"] = tokenized_lbls["input_ids"]
        return tokenized_batch

    def load_tokenized_data(self) -> Tuple[Any, Any]:
        print(f"Loading dataset ({self.config.dataset_id})...")
        dataset = load_dataset(self.config.dataset_id)

        train_set = dataset["train"].map(self._apply_template, fn_kwargs={"mode": "train"})
        tokenized_train_set = train_set.map(self._tokenize_function, batched=True)
        tokenized_train_set.set_format("torch", columns=self.required_columns)

        eval_set = dataset["validation"].map(self._apply_template, fn_kwargs={"mode": "test"})
        tokenized_eval_set = eval_set.map(self._tokenize_function, batched=True)
        tokenized_eval_set.set_format("torch", columns=self.required_columns)

        return tokenized_train_set.select(range(1000)), tokenized_eval_set.select(range(100))
