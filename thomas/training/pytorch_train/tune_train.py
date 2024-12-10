# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torchtune.models.llama3_2 import lora_llama3_2_1b
from torchtune.modules.peft._utils import get_adapter_params, set_trainable_params

from thomas.training.pytorch_train.consts import CONFIG_PATH, TrainConfig
from thomas.training.pytorch_train.utils import load_data, print_trainable_params
from thomas.tooling.cli import generate_config


if __name__ == "__main__":
    # Load config
    config = generate_config(TrainConfig, CONFIG_PATH)

    # Init model
    model = lora_llama3_2_1b(
        lora_attn_modules=["q_proj", "v_proj"], lora_rank=config.lora_r, lora_alpha=config.lora_alpha
    )
    lora_params = get_adapter_params(model)
    # Set requires_grad=True on lora_params, and requires_grad=False on all others.
    set_trainable_params(model, lora_params)

    model.to(config.device)

    # Print the total number of parameters
    print_trainable_params(model)

    # Load dataset
    train_dataloader, _, _, _ = load_data(config)

    # Train loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    step_counter = 0
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader):
            batch = {k: v.to(config.device) for k, v in batch.items()}

            outputs = model(batch["input_ids"])

            loss = loss_fn(outputs.flatten(0, 1), batch["labels"].flatten())
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            step_counter += 1

            if step_counter % config.save_steps == 0:
                torch.save(
                    {
                        "step": step_counter,
                        "model_state_dict": model.state_dict(),
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
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_epoch_loss,
            "ppl": train_ppl,
        },
        os.path.join(config.output_path, f"final_model.pt"),
    )
