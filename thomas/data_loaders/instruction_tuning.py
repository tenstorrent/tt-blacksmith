# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List
from string import Template

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from pydantic import BaseModel


PROMPT_COLUMN = "prompt"
REQUIRED_COLUMNS = ["input_ids", "attention_mask", "labels"]

ALPACA_TEMPLATE = Template(
    """
### Instruction:
$instruction

### Input:
$input

### Response:
"""
)


class InstructionTuningDataLoadingConfig(BaseModel):
    dataset_id: str
    cache_dir: str
    max_length: int
    instruction_columns: List[str]
    batch_size: int
    train_sample: int
    validation_sample: int


class InstructionTuningDataStore:
    def __init__(self, config: InstructionTuningDataLoadingConfig, model_id: str):
        self.dataset = load_dataset(config.dataset_id, cache_dir=config.cache_dir)
        self.batch_size = config.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        self._prep_examples(config.instruction_columns, config.max_length)

        ds = self.tokenized_dataset.train_test_split(test_size=0.2)
        self.train_set = ds["train"].select(range(config.train_sample))
        self.validation_set = ds["test"].select(range(config.validation_sample))

    def _prep_examples(self, instruction_columns: List[str], max_length: int):
        self.dataset = self.dataset.map(
            self._prep_instructions, batched=True, fn_kwargs={"instruction_columns": instruction_columns}
        )

        def _tokenize_function(example: dict, max_length: int):
            tokenized_batch = self.tokenizer(
                example[PROMPT_COLUMN], padding="max_length", max_length=max_length, truncation=True
            )
            tokenized_labels = self.tokenizer(
                example["output"], padding="max_length", max_length=max_length, truncation=True
            )
            tokenized_batch["labels"] = tokenized_labels["input_ids"]

            return tokenized_batch

        self.tokenized_dataset = self.dataset["train"].map(
            _tokenize_function, batched=True, fn_kwargs={"max_length": max_length}
        )

        remove_columns = [col for col in self.tokenized_dataset.column_names if col not in REQUIRED_COLUMNS]
        self.tokenized_dataset = self.tokenized_dataset.remove_columns(remove_columns)

        self.tokenized_dataset.set_format("torch", columns=REQUIRED_COLUMNS)

    def _prep_instructions(self, example: dict, instruction_columns: List[str]):
        example[PROMPT_COLUMN] = [
            ALPACA_TEMPLATE.substitute(instruction=instruction, input=input)
            for instruction, input in zip(example[instruction_columns[0]], example[instruction_columns[1]])
        ]

        return example

    def get_data_loaders(self):
        train_dataloader = DataLoader(
            self.train_set,
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            drop_last=True,
        )

        validation_dataloader = DataLoader(
            self.validation_set,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.batch_size,
            drop_last=True,
        )

        return train_dataloader, validation_dataloader
