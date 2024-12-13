# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from pydantic import BaseModel


class LoraDataLoadingConfig(BaseModel):
    dataset_id: str
    max_length: int
    batch_size: int
    train_sample: int
    validation_sample: int


def load_data(config: LoraDataLoadingConfig, model_id: str):
    ds = load_dataset(config.dataset_id)

    def tokenize_function(example: dict, max_length: int):
        return tokenizer(example["text"], truncation=True, padding=True, max_length=max_length)

    def add_labels(example: dict):
        labels = example["input_ids"].copy()
        labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
        return {"labels": labels}

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = ds.map(tokenize_function, batched=True, fn_kwargs={"max_length": config.max_length})
    tokenized_ds = tokenized_ds.map(add_labels)
    tokenized_ds = tokenized_ds.remove_columns(["text"])
    tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_sample = tokenized_ds["train"].select(range(config.train_sample))
    validation_sample = tokenized_ds["test"].select(range(config.validation_sample))

    return train_sample, validation_sample, data_collator, tokenizer
