# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

from tqdm import tqdm

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from blacksmith.experiments.torch.qwen.configs import TrainingConfig
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.cli import generate_config
from blacksmith.datasets.torch.text2sql.text2sql_dataset import TextToSQLDataset
from blacksmith.tools.reproducibility_manager import ReproducibilityManager


def inspect_data_batches(tokenizer, batch):
    for ind in range(len(batch["input_ids"])):
        full_text = tokenizer.decode(batch["input_ids"][ind], skip_special_tokens=False)

        labels_for_decoding = batch["labels"][ind].clone()
        labels_for_decoding[labels_for_decoding == -100] = tokenizer.pad_token_id
        target_text = tokenizer.decode(labels_for_decoding, skip_special_tokens=False)

        print(f"Full text: {full_text}")
        print(f"Target text: {target_text}")
        print(f"Labels: {batch['labels'][ind]}")
        print(f"Num of tokens: {len(batch['input_ids'][ind])}")
        print("-" * 80)

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Config setup
    config_file_path = os.path.join(os.path.dirname(__file__), "test_qwen_finetuning.yaml")
    config = generate_config(TrainingConfig, config_file_path)

    # Reproducibility setup
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Device setup
    xr.runtime.set_device_type("TT")
    device = xm.xla_device()

    # Load model
    model = get_model(config)
    model = model.to(device)

    # Load dataset
    dataset = TextToSQLDataset(config=config)
    train_dataloader = dataset.get_dataloader()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for epoch in range(config.num_epochs):
        model.train()

        for ind, batch in enumerate(tqdm(train_dataloader)):
            # inspect_data_batches(dataset.tokenizer, batch)

            optimizer.zero_grad()

            # move tensors to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            print(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

            # xm.optimizer_step(optimizer)
            torch_xla.sync(wait=True)

            if ind == 10:
                break
