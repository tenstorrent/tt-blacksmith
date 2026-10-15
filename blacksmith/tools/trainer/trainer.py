# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Union

from tqdm import tqdm

from blacksmith.configs.checkpoint import CheckpointConfig
from blacksmith.configs.trainer import TrainerConfig
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager
from blacksmith.tools.trainer.callback import Callback
from blacksmith.tools.trainer.callbacks_handler import CallbackHandler
from blacksmith.tools.trainer.utils import normalize_callbacks


class Trainer(ABC):
    """Framework-agnostic training loop.

    This base owns the loop, the callback dispatch, gradient-accumulation
    bookkeeping and validation scheduling, and must stay importable without a
    deep-learning framework (no ``torch``). Everything that touches tensors /
    autograd / devices is expressed as an abstract hook or a manager factory,
    implemented by a concrete subclass such as ``CrankLLMTrainer``.
    """

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
        # Loss accumulator over a gradient-accumulation window. Kept as a detached
        # (device) loss object, never read back to the host here: materializing it
        # to a float every step would force a device sync and break the compiled
        # graph. Consumers (e.g. MetricsCallback) convert to a float off the hot path.
        self.step_loss: Any = None

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
                run_config=config.model_dump(mode="json"),
            )
        if self.reproducibility_manager is None:
            self.reproducibility_manager = self._create_reproducibility_manager()
        self.reproducibility_manager.setup()
        self.device_manager = self._create_device_manager()

        self.model = self._load_model()
        # Once, here: DTensor parameters stay DTensors, and the optimizer below must see them.
        self.model = self.device_manager.shard_model(self.model)
        self.train_dataloader, self.val_dataloader = self._load_dataloaders()
        self.optimizer = self._load_optimizer()

    # --------------------------------------------------------- manager factories
    @abstractmethod
    def _create_reproducibility_manager(self) -> ReproducibilityManager:
        """Build the reproducibility manager for this trainer's framework."""
        pass

    @abstractmethod
    def _create_device_manager(self) -> DeviceManager:
        """Build the device manager for this trainer's framework."""
        pass

    @abstractmethod
    def create_checkpoint_manager(
        self,
        checkpoint_config: CheckpointConfig,
        logger: TrainingLogger | None = None,
    ) -> CheckpointManager:
        """Build a checkpoint manager for this trainer's framework.

        Used by the checkpoint callback so it stays free of framework imports.
        """
        pass

    # ------------------------------------------------------------------- hooks
    @abstractmethod
    def _load_model(self) -> Any:
        """
        Build and return the model to train.
        """
        pass

    @abstractmethod
    def _load_dataloaders(self) -> tuple[Iterable, Iterable | None]:
        """
        Build and return the ``(train_dataloader, val_dataloader)`` pair.

        ``val_dataloader`` may be ``None`` if there is no validation set.
        """
        pass

    @abstractmethod
    def _load_optimizer(self) -> Any:
        """Build and return the optimizer over the model's trainable parameters."""
        pass

    @contextmanager
    def _train_lifecycle(self) -> Iterator[None]:
        """Run error / end hooks around the training loop body.

        An exception is passed to ``on_error`` and propagates only if a callback re-raises it.
        ``on_train_end`` fires only when the loop completes, so it never sees a half-trained model.
        """
        try:
            yield
        except Exception as exception:
            self.callback_handler("on_error", exception)
        else:
            self.callback_handler("on_train_end")
        finally:
            # Finish the logger after the callbacks so they can still use it.
            if self.logger is not None:
                self.logger.finish()

    def _init_step_state(self) -> None:
        """Reset the loss accumulator at the start of ``train()``."""
        self.step_loss = None

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

                self._train_mode()
                # Drop a trailing partial window from the previous epoch so its
                # gradients and loss do not leak into this epoch's first step.
                accumulation_step = 0
                self.step_loss = None
                self._zero_grad()

                progress = tqdm(self.train_dataloader, desc=f"Training (epoch {epoch})")
                for batch in progress:
                    self.callback_handler("on_train_batch_start", batch)

                    batch = self.device_manager.prepare_batch(batch)

                    self.callback_handler("on_forward_start", batch)
                    loss = self._forward(batch)
                    self.callback_handler("on_forward_end", loss)

                    # Scale here so _forward stays shared with val.
                    scaled_loss = loss / grad_accumulation_steps
                    self.callback_handler("on_backward_start", loss)
                    self._backward(scaled_loss)
                    # Accumulate the detached loss on the device; no host readback here.
                    self.step_loss = self._accumulate_loss(self.step_loss, scaled_loss)
                    self.callback_handler("on_backward_end", loss)

                    accumulation_step += 1

                    if accumulation_step == grad_accumulation_steps:
                        window_loss = self.step_loss
                        self.step_loss = None
                        self.callback_handler("on_optimizer_step_start")
                        self._optimizer_step()
                        self.callback_handler("on_optimizer_step_end", window_loss)

                        accumulation_step = 0
                        self.global_step += 1

                        # Periodic inline validation.
                        if self._validation_enabled() and self.global_step % self.config.val_steps_freq == 0:
                            self.validate()

                        self.callback_handler("on_step_end", self.global_step)

                    self.callback_handler("on_train_batch_end")

                self.callback_handler("on_train_epoch_end")

    def _validation_enabled(self) -> bool:
        return self.val_dataloader is not None and self.config.val_steps_freq > 0

    def validate(self) -> None:
        self._eval_mode()
        self.callback_handler("on_validation_start")

        total_loss = None
        num_batches = 0
        with self._no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                self.callback_handler("on_validation_batch_start", batch)

                batch = self.device_manager.prepare_batch(batch)
                loss = self._forward(batch)

                # Accumulate the detached loss on the device; no host readback here.
                total_loss = self._accumulate_loss(total_loss, loss)
                num_batches += 1
                self.callback_handler("on_validation_batch_end", batch, loss)

        # Device-side mean (or 0.0 when there is no validation data). Consumers
        # read it back to the host off the hot path.
        val_loss = total_loss / num_batches if num_batches else 0.0
        self.callback_handler("on_validation_end", val_loss)
        self._train_mode()

    @abstractmethod
    def _forward(self, batch: Any) -> Any:
        """
        Run a forward pass for a single batch and return the raw (unscaled) loss.

        Shared by both ``train`` and ``validate``; gradient-accumulation scaling
        is applied by ``train`` before backward.
        """
        pass

    @abstractmethod
    def _backward(self, loss: Any) -> None:
        """Run the backward pass for ``loss``."""
        pass

    def _optimizer_step(self) -> None:
        """
        Step the optimizer and reset gradients. Override for custom behaviour.
        """
        self.device_manager.optimizer_step(self.optimizer, zero_grad=True)

    @abstractmethod
    def _zero_grad(self) -> None:
        """Reset accumulated gradients without stepping the optimizer."""
        pass

    @abstractmethod
    def _train_mode(self) -> None:
        """Put the model in training mode."""
        pass

    @abstractmethod
    def _eval_mode(self) -> None:
        """Put the model in evaluation mode."""
        pass

    @abstractmethod
    def _no_grad(self) -> Any:
        """Return a context manager that disables gradient tracking."""
        pass

    @abstractmethod
    def _accumulate_loss(self, running: Any, loss: Any) -> Any:
        """Add ``loss`` to the running accumulator, staying on the device.

        Must detach ``loss`` (so the autograd graph is not retained) but must not
        read it back to the host: pulling the value to a float here would force a
        device sync every step and break the compiled graph. ``running`` is
        ``None`` for the first loss of a window.
        """
        pass

    @abstractmethod
    def loss_to_float(self, loss: Any) -> float:
        """Read a loss back to the host as a Python float.

        Must reduce a sharded / partial loss across devices first, so every rank
        reports the global value rather than its local shard. ``loss`` may also
        be a plain number (e.g. ``0.0`` from an empty validation set).
        """
        pass

    @abstractmethod
    def parameter_counts(self) -> tuple[int, int]:
        """Return ``(total_params, trainable_params)`` for the current model."""
        pass

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
