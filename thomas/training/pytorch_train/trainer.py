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

        # self.save_period = cfg_trainer['save_period']
        # self.metric_ftns = metric_ftns

        # configuration to monitor model performance and save best
        # if self.monitor == 'off':
        #     self.mnt_mode = 'off'
        #     self.mnt_best = 0
        # else:
        #     self.mnt_mode, self.mnt_metric = self.monitor.split()
        #     assert self.mnt_mode in ['min', 'max']

    #
    #     self.mnt_best = inf if self.mnt_mode == 'min' else -inf
    #     self.early_stop = cfg_trainer.get('early_stop', inf)
    #     if self.early_stop <= 0:
    #         self.early_stop = inf
    #
    # self.checkpoint_dir = config.save_dir
    #
    # if config.resume is not None:
    #     self._resume_checkpoint(config.resume)

    def _train_epoch(self, epoch):
        self.model.train()

        epoch_loss = 0
        for _, batch in enumerate(tqdm(self.train_dataloader)):
            batch = {k: v.to(self.run_on) for k, v in batch.items()}

            self.optimizer.zero_grad()

            logits = self.model(**batch).logits

            loss = self.loss(logits.flatten(0, 1), batch["labels"].flatten())
            epoch_loss += loss.detach().item()
            loss.backward()
            self.optimizer.step()

        epoch_loss /= len(self.train_dataloader)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.do_validation:
            validation_loss = self._valid_epoch()

            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            # self.train_metrics.update('loss', loss.item())
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(output, target))
        #
        # if batch_idx % self.log_step == 0:
        #     self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
        #         epoch,
        #         self._progress(batch_idx),
        #         loss.item()))
        #     self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        #
        # if batch_idx == self.len_epoch:
        #     break
        # log = self.train_metrics.result()

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
            epoch_loss, validation_loss = self._train_epoch(epoch)
            losses.append((epoch_loss, validation_loss))

            # if epoch % self.save_period == 0:
            #     self._save_checkpoint(epoch, save_best=best)

        for idx, loss in enumerate(losses, start=1):
            print(f"Epoch {idx}, Loss: {loss[0]}, Validation Loss: {loss[1]}")

            # Logging and evaluation step

            # save logged informations into log dict
            # log = {'epoch': epoch, 'epoch_loss': epoch_loss}
            # log.update(result)


#
# # print logged informations to the screen
# for key, value in log.items():
#     self.logger.info('    {:15s}: {}'.format(str(key), value))
#
# # evaluate model performance according to configured metric, save best checkpoint as model_best
# best = False
# if self.mnt_mode != 'off':
#     try:
#         # check whether model performance improved or not, according to specified metric(mnt_metric)
#         improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
#                    (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
#     except KeyError:
#         self.logger.warning("Warning: Metric '{}' is not found. "
#                             "Model performance monitoring is disabled.".format(self.mnt_metric))
#         self.mnt_mode = 'off'
#         improved = False

# Early stopping logic
# if improved:
#     self.mnt_best = log[self.mnt_metric]
#     not_improved_count = 0
#     best = True
# else:
#     not_improved_count += 1
#
# if not_improved_count > self.early_stop:
#     self.logger.info("Validation performance didn\'t improve for {} epochs. "
#                      "Training stops.".format(self.early_stop))
#     break
