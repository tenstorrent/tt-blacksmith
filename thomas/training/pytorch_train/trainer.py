# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from tqdm import tqdm
import torch
from transformers import get_linear_schedule_with_warmup


class PyTorchTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader, config):
        self.model = model

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.loss = config.loss()
        self.optimizer = config.optimizer(self.model.parameters(), **config.optimizer_kwargs)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_dataloader) * config.epochs),
        )
        self.epochs = config.epochs
        self.start_epoch = 1
        self.run_on = config.run_on
        self.do_validation = config.do_validation

    def _train_epoch(self):
        self.model.train()

        epoch_loss = 0
        for _, batch in enumerate(tqdm(self.train_dataloader)):
            batch = {k: v.to(self.run_on) for k, v in batch.items()}

            self.optimizer.zero_grad()

            # logits = self.model(**batch).logits
            logits = self.model(batch["input_ids"])

            loss = self.loss(logits.flatten(0, 1), batch["labels"].flatten())
            epoch_loss += loss.detach().item()
            loss.backward()
            self.optimizer.step()

        epoch_loss /= len(self.train_dataloader)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.do_validation:
            validation_loss = self._valid_epoch()

        return epoch_loss, validation_loss

    def _valid_epoch(self):
        self.model.eval()

        with torch.no_grad():
            validation_loss = 0
            for _, batch in enumerate(tqdm(self.valid_dataloader)):
                batch = {k: v.to(self.run_on) for k, v in batch.items()}

                logits = self.model(**batch).logits

                loss = self.loss(logits.flatten(0, 1), batch["labels"].flatten())
                validation_loss += loss.detach().item()

        validation_loss /= len(self.valid_dataloader)

        return validation_loss

    def train(self):
        """
        Full training logic
        """
        losses = []
        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_loss, validation_loss = self._train_epoch()
            losses.append((epoch_loss, validation_loss))

        for idx, loss in enumerate(losses, start=1):
            print(f"Epoch {idx}, Loss: {loss[0]}, Validation Loss: {loss[1]}")

            # Logging and evaluation step
