# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from thomas.training.pytorch_train.consts import CONFIG_PATH, TrainConfig
from thomas.training.pytorch_train.utils import load_data, init_hf_model
from thomas.tooling.cli import generate_config


if __name__ == "__main__":
    # Load config
    config = generate_config(TrainConfig, CONFIG_PATH)

    # Init model
    model = init_hf_model(config)

    # Load dataset
    train_dataloader, _, _, _ = load_data(config)

    # Train loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    step_counter = 0
    for epoch in range(config.num_epochs):
        model.train()

        epoch_loss = 0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            batch = {k: v.to(config.device) for k, v in batch.items()}

            outputs = model(**batch)

            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            step_counter += 1

            if step_counter % config.save_steps == 0:
                model.save_pretrained(os.path.join(config.output_path, f"checkpoint_{step_counter}"))

        train_epoch_loss = epoch_loss / len(train_dataloader)
        train_ppl = torch.exp(torch.tensor([train_epoch_loss]))
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

    model.save_pretrained(os.path.join(config.output_path, "final_model"))
