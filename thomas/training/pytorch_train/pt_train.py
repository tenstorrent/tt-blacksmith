# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import yaml

from thomas.training.pytorch_train.consts import CONFIG_PATH


def tokenize_function(example: dict, max_length: int):
    return tokenizer(example["text"], truncation=True, padding=True, max_length=max_length)


def add_labels(example: dict):
    labels = example["input_ids"].copy()
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
    return {"labels": labels}


if __name__ == "__main__":
    # Load config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Init model
    model = AutoModelForCausalLM.from_pretrained(config["model_id"], torch_dtype=torch.float16)
    lora_config = LoraConfig(r=2, lora_alpha=16)
    model = get_peft_model(model, lora_config)
    model.to(config["device"])
    model.print_trainable_parameters()

    # Load dataset
    ds = load_dataset(config["dataset_id"])

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = ds.map(tokenize_function, batched=True, fn_kwargs={"max_length": config["max_length"]})
    tokenized_ds = tokenized_ds.map(add_labels)
    tokenized_ds = tokenized_ds.remove_columns(["text"])
    tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_sample = tokenized_ds["train"].select(range(100))
    train_dataloader = DataLoader(
        train_sample, shuffle=True, collate_fn=data_collator, batch_size=config["batch_size"], pin_memory=True
    )

    # Train loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]))
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * config["num_epochs"]),
    )

    step_counter = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(config["device"]) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            step_counter += 1

            if step_counter % config["save_steps"] == 0:
                model.save_pretrained(os.path.join(config["output_path"], f"checkpoint_{step_counter}"))

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

    model.save_pretrained(os.path.join(config["output_path"], "final_model"))
