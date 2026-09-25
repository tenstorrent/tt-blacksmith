# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from blacksmith.tools.crank.device_manager import DeviceManager
from blacksmith.tools.crank.logging_manager import TrainingLogger
from blacksmith.tools.crank.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.crank.torch_helpers import loss_to_float
from blacksmith.tools.crank.trainer.callback import Callback
from blacksmith.tools.crank.trainer.callbacks_handler import CallbackHandler
from blacksmith.tools.crank.trainer.configs.base import TrainerConfig
from blacksmith.tools.crank.trainer.utils import normalize_callbacks


class Trainer(ABC):
    def __init__(
        self,
        callbacks: Union[Callback, Sequence[Callback], None] = None,
    ):
        self.callback_handler = CallbackHandler(self, normalize_callbacks(callbacks))
        self.config: TrainerConfig | None = None
        self.reproducibility_manager: ReproducibilityManager | None = None
        # The trainer owns the logger; callbacks (and other consumers) use it via
        # `trainer.logger`. It is created in `setup` and finished in `train`.
        self.logger: TrainingLogger | None = None
        self.global_step: int = 0
        self.epoch: int = 0
        # Host-side loss accumulator over a gradient-accumulation window.
        self.step_loss: torch.Tensor | None = None

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
        if config is not None:
            self.logger = TrainingLogger(
                config.logging,
                kwargs.get("test_log_filename_prefix"),
            )
        if self.reproducibility_manager is None:
            self.reproducibility_manager = ReproducibilityManager(config)
        self.reproducibility_manager.setup()
        self.device_manager = DeviceManager(config)

        self.model = self._load_model()
        # Once, here: DTensor parameters stay DTensors, and the optimizer below must see them.
        self.model = self.device_manager.shard_model(self.model)
        self.train_dataloader, self.val_dataloader = self._load_dataloaders()
        self.optimizer = self._load_optimizer()

    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        """
        Build and return the model to train.
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
        )

    @contextmanager
    def _train_lifecycle(self) -> Iterator[None]:
        """Run error / cleanup hooks around the training loop body."""
        try:
            yield
        except Exception as exception:
            self.callback_handler("on_error", exception)
        finally:
            self.callback_handler("on_train_end")
            # Finish the logger after `on_train_end` so callbacks can still use it.
            if self.logger is not None:
                self.logger.finish()

    def _init_step_state(self) -> None:
        """Reset the host-side loss accumulator at the start of ``train()``."""
        self.step_loss = torch.zeros(())

    def train(self) -> None:
        self.callback_handler("on_train_start")
        # `on_train_start` callbacks need to guard against having a config.
        if self.config is None:
            return

        grad_accumulation_steps = self.config.gradient_accumulation_steps
        # Multichip: plain tensors HF builds internally count as replicated DTensors.
        with self._train_lifecycle(), self.device_manager.replication_context():
            self._init_step_state()

            # Initial validation pass before any optimizer steps.
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

                        # Periodic inline validation.
                        if self._validation_enabled() and self.global_step % self.config.val_steps_freq == 0:
                            self.validate()

                    self.callback_handler("on_train_batch_end")

                self.callback_handler("on_train_epoch_end")

    def _validation_enabled(self) -> bool:
        return self.val_dataloader is not None and self.config.val_steps_freq > 0

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
        self.device_manager.optimizer_step(self.optimizer, zero_grad=True)

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
        self.logger = None
        self.global_step = 0
        self.epoch = 0
        self.step_loss = None
