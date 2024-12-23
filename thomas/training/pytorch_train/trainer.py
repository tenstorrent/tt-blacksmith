# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from tqdm import tqdm
import torch
from transformers import get_linear_schedule_with_warmup
from peft.peft_model import PeftModel


class PyTorchTrainer:
    def __init__(self, model, train_dataloader, valid_dataloader, config, logger):
        self.model = model

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.loss = config.loss()
        self.optimizer = config.optimizer(self.model.parameters(), **config.optimizer_kwargs)
        self.lr_scheduler = None
        self.epochs = config.epochs
        self.start_epoch = 1
        self.global_step = 0
        self.run_on = config.run_on
        self.do_validation = config.do_validation

        self.logger = logger

    def _train_batch(self, batch):
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y)
        return loss

    def _prepare_batch(self, batch):
        return batch

    def _train_epoch(self):
        self.model.train()

        epoch_loss = 0
        for _, batch in enumerate(tqdm(self.train_dataloader)):
            self.global_step += 1
            batch = self._prepare_batch(batch)

            self.optimizer.zero_grad()
            loss = self._train_batch(batch)

            epoch_loss += loss.item()
            self.logger.log_metrics({"batch_loss": loss.item()}, step=self.global_step)
            self.logger.save_checkpoint_step(self.model, self.optimizer, self.global_step)

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
                validation_loss += loss.item()

        validation_loss /= len(self.valid_dataloader)
        self.logger.log_metrics({"validation_loss": validation_loss})

        return validation_loss

    def train(self):
        self.logger.watch_model(self.model, log="all", log_freq=self.logger.config.log_every_n_steps)

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_loss, validation_loss = self._train_epoch()

            self.logger.log_metrics({"epoch_loss": epoch_loss, "epoch": epoch})
            self.logger.save_checkpoint_epoch(self.model, self.optimizer, epoch, epoch_loss, validation_loss)


class LoraTrainer(PyTorchTrainer):
    def __init__(self, model, train_dataloader, valid_dataloader, config, logger):
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_dataloader) * config.epochs),
        )
        super().__init__(model, train_dataloader, valid_dataloader, config, logger)

    def _prepare_batch(self, batch):
        run_on = self.run_on
        # HACK: TT doesn't support moving to "tt" device
        if self.run_on == "tt":
            run_on = "cpu"

        batch = {k: v.to(run_on) for k, v in batch.items()}
        return batch

    def _train_batch(self, batch):
        x, y = batch["input_ids"], batch["labels"]
        pred = self.model(x, labels=y)
        return pred.loss
