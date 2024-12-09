# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

from thomas.training.pytorch_train.consts import CONFIG_PATH, TrainConfig
from thomas.tooling.cli import generate_config


def tokenize_function(example: dict, max_length: int):
    return tokenizer(example["text"], truncation=True, padding=True, max_length=max_length)


def add_labels(example: dict):
    labels = example["input_ids"].copy()
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
    return {"labels": labels}


if __name__ == "__main__":
    # Load config
    config = generate_config(TrainConfig, CONFIG_PATH)

    # Init model
    model = AutoModelForCausalLM.from_pretrained(config.model_id, torch_dtype=torch.float16)
    lora_config = LoraConfig(r=config.lora_r, lora_alpha=config.lora_alpha)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    ds = load_dataset(config.dataset_id)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = ds.map(tokenize_function, batched=True, fn_kwargs={"max_length": config.max_length})
    tokenized_ds = tokenized_ds.map(add_labels)
    tokenized_ds = tokenized_ds.remove_columns(["text"])
    tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Start training
    train_args = TrainingArguments(
        output_dir=config.output_path,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        remove_unused_columns=False,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=config.save_steps,
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=tokenized_ds["train"].select(range(100)),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
