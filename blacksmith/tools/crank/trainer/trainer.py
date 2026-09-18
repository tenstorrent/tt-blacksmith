# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tt-crank `Trainer`.

Subclass of the tt-xla `Trainer` that swaps out what depends on the backend and keeps
the rest (callbacks, lifecycle, `cleanup`, the abstract strategy hooks). What changes:

- `setup` uses the tt-crank `DeviceManager` and shards the model *before* the
  optimizer is built: `shard_model` replaces parameters with DTensors, and the
  optimizer has to hold those (tt-xla annotated in place, so order did not matter).
- The training loop is eager. No `torch_xla.sync()` fences, no `capturable=True`,
  no pre-seeded grads / AdamW moments; `step_loss` is a host-side scalar so
  callbacks can keep calling `.item()` on `window_loss` (under data parallelism the
  device loss is a `Partial` DTensor, which `loss_to_float` reduces).
- Compile options are per `torch.compile` call (strategies read
  `self.device_manager.compile_options()`), so `_apply_tt_compile_options` is a no-op.
"""
from typing import Any

import torch
from tqdm import tqdm

from blacksmith.tools.crank.device_manager import DeviceManager
from blacksmith.tools.crank.torch_helpers import loss_to_float
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.trainer.configs.base import TrainerConfig
from blacksmith.tools.trainer.trainer import Trainer as XlaTrainer


class Trainer(XlaTrainer):
    def setup(self, config: TrainerConfig | None = None, **kwargs: Any) -> None:
        if self.config is not None:
            self.cleanup()

        self.config = config
        if config is not None:
            self.logger = TrainingLogger(config.logging, kwargs.get("test_log_filename_prefix"))
        if self.reproducibility_manager is None:
            self.reproducibility_manager = ReproducibilityManager(config)
        self.reproducibility_manager.setup()
        self.device_manager = DeviceManager(config)

        self.model = self._load_model()
        # Once, here: DTensor parameters stay DTensors, and the optimizer below must see them.
        self.model = self.device_manager.shard_model(self.model)
        self.train_dataloader, self.val_dataloader = self._load_dataloaders()
        self.optimizer = self._load_optimizer()

    def _load_optimizer(self) -> torch.optim.Optimizer:
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        # No `capturable`: that kept the tt-xla fused step graph stable; tt-crank is eager.
        return torch.optim.AdamW(trainable_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _apply_tt_compile_options(self) -> None:
        # Per `torch.compile` call on tt-crank; see `DeviceManager.compile_options()`.
        return

    def _init_fused_step_state(self) -> None:
        # Host-side accumulator over a gradient-accumulation window; nothing to pre-seed.
        self.step_loss = torch.zeros(())

    def train(self) -> None:
        self.callback_handler("on_train_start")
        if self.config is None:
            return

        grad_accumulation_steps = self.config.gradient_accumulation_steps
        with self._train_lifecycle(), self.device_manager.replication_context():
            self._init_fused_step_state()

            if self._validation_enabled():
                self.validate()

            for epoch in range(self.config.num_epochs):
                self.epoch = epoch
                self.callback_handler("on_train_epoch_start")

                self.model.train()
                accumulation_step = 0

                progress = tqdm(self.train_dataloader, desc=f"Training (epoch {epoch})")
                for batch in progress:
                    self.callback_handler("on_train_batch_start", batch)

                    # Keep ``labels`` on CPU; one-hot on device OOMs (#455).
                    batch = self.device_manager.prepare_batch(batch, skip_keys=("labels",))

                    self.callback_handler("on_forward_start", batch)
                    loss = self._forward(batch)
                    self.callback_handler("on_forward_end", loss)

                    # Scale here so _forward stays shared with val.
                    scaled_loss = loss / grad_accumulation_steps
                    self.callback_handler("on_backward_start", loss)
                    self._backward(scaled_loss)
                    self.step_loss = self.step_loss + loss_to_float(scaled_loss)
                    self.callback_handler("on_backward_end", loss)

                    accumulation_step += 1

                    if accumulation_step == grad_accumulation_steps:
                        window_loss = self.step_loss
                        self.step_loss = torch.zeros(())
                        self.callback_handler("on_optimizer_step_start")
                        self._optimizer_step()
                        self.callback_handler("on_optimizer_step_end", window_loss)

                        accumulation_step = 0
                        self.global_step += 1
                        progress.set_postfix(loss=window_loss.item())

                        if self._validation_enabled() and self.global_step % self.config.val_steps_freq == 0:
                            self.validate()

                    self.callback_handler("on_train_batch_end")

                self.callback_handler("on_train_epoch_end")

    def validate(self) -> None:
        self.model.eval()
        self.callback_handler("on_validation_start")

        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                self.callback_handler("on_validation_batch_start", batch)

                # Keep ``labels`` on CPU; one-hot on device OOMs (#455).
                batch = self.device_manager.prepare_batch(batch, skip_keys=("labels",))
                loss = self._forward(batch)

                total_loss += loss_to_float(loss)
                num_batches += 1
                self.callback_handler("on_validation_batch_end", batch, loss)

        val_loss = total_loss / num_batches if num_batches else 0.0
        self.callback_handler("on_validation_end", val_loss)
        self.model.train()
