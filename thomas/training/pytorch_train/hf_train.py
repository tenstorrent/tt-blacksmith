# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from transformers import TrainingArguments, Trainer

from thomas.training.pytorch_train.consts import CONFIG_PATH, TrainConfig
from thomas.training.pytorch_train.utils import init_hf_model, load_data
from thomas.tooling.cli import generate_config


if __name__ == "__main__":
    # Load config
    config = generate_config(TrainConfig, CONFIG_PATH)

    # Init model
    model = init_hf_model(config)

    # Load dataset
    _, data_collator, train_dataset, tokenizer = load_data(config)

    # Start training
    train_args = TrainingArguments(
        output_dir=config.output_path,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        remove_unused_columns=False,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=config.save_steps,
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
