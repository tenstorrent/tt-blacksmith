# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

from thomas.training.pytorch_train.consts import TrainConfig


def load_data(config: TrainConfig):
    ds = load_dataset(config.dataset_id)

    def tokenize_function(example: dict, max_length: int):
        return tokenizer(example["text"], truncation=True, padding=True, max_length=max_length)

    def add_labels(example: dict):
        labels = example["input_ids"].copy()
        labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
        return {"labels": labels}

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = ds.map(tokenize_function, batched=True, fn_kwargs={"max_length": config.max_length})
    tokenized_ds = tokenized_ds.map(add_labels)
    tokenized_ds = tokenized_ds.remove_columns(["text"])
    tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_sample = tokenized_ds["train"].select(range(100))
    train_dataloader = DataLoader(
        train_sample, shuffle=True, collate_fn=data_collator, batch_size=config.batch_size, pin_memory=True
    )

    return train_dataloader, data_collator, train_sample, tokenizer


def init_hf_model(config: TrainConfig):
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=torch.float16)
    lora_config = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha)
    model = get_peft_model(model, lora_config)
    model.to(config.device)

    print_trainable_params(model)

    return model


def print_trainable_params(model):
    total_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(
        f"""
    {total_params} total params,
    {trainable_params}" trainable params,
    {(100.0 * trainable_params / total_params):.2f}% of all params are trainable.
    """
    )
