# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
from tqdm import tqdm
from torchtune.models.llama3_2 import lora_llama3_2_1b
from torchtune.modules.peft._utils import get_adapter_params, set_trainable_params

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
    lora_model = lora_llama3_2_1b(
        lora_attn_modules=["q_proj", "v_proj"], lora_rank=config.lora_r, lora_alpha=config.lora_alpha
    )
    lora_params = get_adapter_params(lora_model)
    # Set requires_grad=True on lora_params, and requires_grad=False on all others.
    set_trainable_params(lora_model, lora_params)

    lora_model.to(config.device)

    # Print the total number of parameters
    total_params = sum([p.numel() for p in lora_model.parameters()])
    trainable_params = sum([p.numel() for p in lora_model.parameters() if p.requires_grad])
    print(
        f"""
    {total_params} total params,
    {trainable_params}" trainable params,
    {(100.0 * trainable_params / total_params):.2f}% of all params are trainable.
    """
    )

    # Load dataset
    ds = load_dataset(config.dataset_id)

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

    # Train loop
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=config.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    step_counter = 0
    for epoch in range(config.num_epochs):
        lora_model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(config.device) for k, v in batch.items()}

            outputs = lora_model(batch["input_ids"])

            loss = loss_fn(outputs.flatten(0, 1), batch["labels"].flatten())
            total_loss += loss.detach().item()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            step_counter += 1

            if step_counter % config.save_steps == 0:
                torch.save(
                    {
                        "step": step_counter,
                        "model_state_dict": lora_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    os.path.join(config.output_path, f"checkpoint_{step_counter}.pt"),
                )

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(torch.tensor([train_epoch_loss]))
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

    torch.save(
        {
            "step": step_counter,
            "model_state_dict": lora_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_epoch_loss,
            "ppl": train_ppl,
        },
        os.path.join(config.output_path, f"final_model.pt"),
    )
