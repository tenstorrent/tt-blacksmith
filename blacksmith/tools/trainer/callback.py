# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC
from enum import StrEnum


class CallbackEvent(StrEnum):
    ON_TRAIN_START = "on_train_start"
    ON_TRAIN_END = "on_train_end"
    ON_TRAIN_EPOCH_START = "on_train_epoch_start"
    ON_TRAIN_EPOCH_END = "on_train_epoch_end"
    ON_TRAIN_BATCH_START = "on_train_batch_start"
    ON_TRAIN_BATCH_END = "on_train_batch_end"
    ON_VALIDATION_START = "on_validation_start"
    ON_VALIDATION_BATCH_START = "on_validation_batch_start"
    ON_VALIDATION_BATCH_END = "on_validation_batch_end"
    ON_VALIDATION_END = "on_validation_end"
    ON_FORWARD_START = "on_forward_start"
    ON_FORWARD_END = "on_forward_end"
    ON_BACKWARD_START = "on_backward_start"
    ON_BACKWARD_END = "on_backward_end"
    ON_OPTIMIZER_STEP_START = "on_optimizer_step_start"
    ON_OPTIMIZER_STEP_END = "on_optimizer_step_end"


class Callback(ABC):
    # Training callbacks.
    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_train_epoch_start(self, trainer):
        pass

    def on_train_epoch_end(self, trainer):
        pass

    def on_train_batch_start(self, trainer, batch):
        pass

    def on_train_batch_end(self, trainer):
        pass

    # Validation callbacks.
    def on_validation_start(self, trainer):
        pass

    def on_validation_batch_start(self, trainer, batch):
        pass

    def on_validation_batch_end(self, trainer, batch):
        pass

    def on_validation_end(self, trainer):
        pass

    # Forward / Backward / Optimizer Step callbacks.
    def on_forward_start(self, trainer, batch):
        pass

    def on_forward_end(self, trainer, loss):
        pass

    def on_backward_start(self, trainer, loss):
        pass

    def on_backward_end(self, trainer):
        pass

    def on_optimizer_step_start(self, trainer):
        pass

    def on_optimizer_step_end(self, trainer):
        pass
