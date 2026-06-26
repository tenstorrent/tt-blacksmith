# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Union

import torch
import torch_xla
from torch.utils.data import DataLoader
from tqdm import tqdm

from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.trainer.callback import Callback
from blacksmith.tools.trainer.callbacks_handler import CallbackHandler
from blacksmith.tools.trainer.configs.base import TrainerConfig
from blacksmith.tools.trainer.utils import normalize_callbacks


class Trainer(ABC):
    def __init__(
        self,
        callbacks: Union[Callback, Sequence[Callback], None] = None,
    ):
        self.callback_handler = CallbackHandler(self, normalize_callbacks(callbacks))
        self.config: TrainerConfig | None = None
        self.reproducibility_manager: ReproducibilityManager | None = None
        self.global_step: int = 0
        self.epoch: int = 0

    def setup(
        self,
        config: TrainerConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Setup the trainer with the given configuration.
        """
        if self.config is not None:
            self.cleanup()

        self.config = config
        if self.reproducibility_manager is None:
            self.reproducibility_manager = ReproducibilityManager(config)
        self.reproducibility_manager.setup()
        self.device_manager = DeviceManager(config)

        self.model = self._load_model()
        self.train_dataloader, self.val_dataloader = self._load_dataloaders()
        self.optimizer = self._load_optimizer()

    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        """
        Build and return compiled model to train.
        """
        pass

    @abstractmethod
    def _load_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        """
        Build and return the ``(train_dataloader, val_dataloader)`` pair.

        ``val_dataloader`` may be ``None`` if there is no validation set.
        """
        pass

    def _load_optimizer(self) -> torch.optim.Optimizer:
        """
        Build and return the optimizer over the model's trainable parameters.

        Defaults to AdamW; override for a different optimizer.
        """
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            capturable=self.config.use_tt,
        )

    def train(self) -> None:
        self.callback_handler("on_train_start")
        # `on_train_start` callbacks need to guard against having a config.
        if self.config is None:
            return

        grad_accumulation_steps = self.config.gradient_accumulation_steps
        # TODO(mmilosevicTT): Temporary running loss for debug prints; remove once
        # default logging callback lands. See https://github.com/tenstorrent/tt-blacksmith/issues/621
        running_loss = 0.0
        running_count = 0
        try:
            # Initial validation pass before any optimizer steps.
            if self.val_dataloader is not None:
                self.validate()

            for epoch in range(self.config.num_epochs):
                self.epoch = epoch
                self.callback_handler("on_train_epoch_start")

                self.model.train()
                self.optimizer.zero_grad()
                accumulation_step = 0

                progress = tqdm(self.train_dataloader, desc=f"Training (epoch {epoch})")
                for batch in progress:
                    self.callback_handler("on_train_batch_start", batch)

                    # Shard inputs (data parallel) and model (tensor parallel) if configured.
                    batch = self.device_manager.prepare_batch(batch)
                    self.device_manager.shard_model(self.model)

                    # Forward.
                    self.callback_handler("on_forward_start", batch)
                    loss = self._forward(batch)
                    self.callback_handler("on_forward_end", loss)

                    # Backward. Gradient-accumulation scaling is applied here so
                    # _forward can stay identical between training and validation.
                    self.callback_handler("on_backward_start", loss)
                    self._backward(loss / grad_accumulation_steps)
                    if self.config.use_tt:
                        torch_xla.sync(wait=True)
                    self.callback_handler("on_backward_end")

                    accumulation_step += 1

                    # Step the optimizer only after accumulating gradients.
                    if accumulation_step == grad_accumulation_steps:
                        self.callback_handler("on_optimizer_step_start")
                        self._optimizer_step()
                        self.callback_handler("on_optimizer_step_end")

                        accumulation_step = 0
                        self.global_step += 1
                        progress.set_postfix(loss=loss.item())
                        # TODO(mmilosevicTT): Temporary debug print (avg over 10 steps); remove once
                        # default logging callback lands. See https://github.com/tenstorrent/tt-blacksmith/issues/621
                        running_loss += loss.item()
                        running_count += 1
                        if self.global_step % 10 == 0:
                            print(f"[step {self.global_step}] loss={running_loss / running_count:.4f}")
                            running_loss = 0.0
                            running_count = 0

                        # Periodic inline validation.
                        if (
                            self.val_dataloader is not None
                            and self.config.val_steps_freq
                            and self.global_step % self.config.val_steps_freq == 0
                        ):
                            self.validate()

                    self.callback_handler("on_train_batch_end")

                self.callback_handler("on_train_epoch_end")
        except Exception as exception:
            self.callback_handler("on_error", trainer=self, exception=exception)
        finally:
            self.callback_handler("on_train_end")

    def validate(self) -> None:
        self.model.eval()
        self.callback_handler("on_validation_start")

        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                self.callback_handler("on_validation_batch_start", batch)

                batch = self.device_manager.prepare_batch(batch)
                self.device_manager.shard_model(self.model)

                loss = self._forward(batch)
                if self.config.use_tt:
                    torch_xla.sync(wait=True)

                total_loss += loss.item()
                num_batches += 1
                self.callback_handler("on_validation_batch_end", batch, loss)

        val_loss = total_loss / num_batches if num_batches else 0.0
        # TODO(mmilosevicTT): Temporary debug print; remove once default logging
        # callback lands. See https://github.com/tenstorrent/tt-blacksmith/issues/621
        print(f"[step {self.global_step}] val_loss={val_loss:.4f}")
        self.callback_handler("on_validation_end", val_loss)
        self.model.train()

    @abstractmethod
    def _forward(self, batch: Any) -> torch.Tensor:
        """
        Run a forward pass for a single batch and return the raw (unscaled) loss.

        Shared by both ``train`` and ``validate``; gradient-accumulation scaling
        is applied by ``train`` before backward.
        """
        pass

    def _backward(self, loss: torch.Tensor) -> None:
        """
        Run the backward pass. Override for custom synchronization behaviour.
        """
        loss.backward()

    def _optimizer_step(self) -> None:
        """
        Step the optimizer and reset gradients. Override for custom behaviour.
        """
        self.device_manager.optimizer_step(self.optimizer)
        self.optimizer.zero_grad()

    def cleanup(self) -> None:
        """
        Clean up trainer state, releasing references to free resources.

        Deletes all state except ``callback_handler`` and resets the trainer to
        a not-set-up state (``config`` is ``None``).
        """
        preserved = {"callback_handler"}
        for attr in list(self.__dict__):
            if attr not in preserved:
                delattr(self, attr)
        self.config = None
        self.reproducibility_manager = None
        self.global_step = 0
        self.epoch = 0
