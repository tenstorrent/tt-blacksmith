# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC


class Callback(ABC):
    # Training callbacks.
    def on_train_start(self, trainer, **kwargs):
        pass

    def on_train_end(self, trainer, **kwargs):
        pass

    def on_train_epoch_start(self, trainer, **kwargs):
        pass

    def on_train_epoch_end(self, trainer, **kwargs):
        pass

    def on_train_batch_start(self, trainer, batch, **kwargs):
        pass

    def on_train_batch_end(self, trainer, **kwargs):
        pass

    # Validation callbacks.
    def on_validation_start(self, trainer, **kwargs):
        pass

    def on_validation_batch_start(self, trainer, batch, **kwargs):
        pass

    def on_validation_batch_end(self, trainer, batch, loss, **kwargs):
        pass

    def on_validation_end(self, trainer, val_loss, **kwargs):
        pass

    # Forward / Backward / Optimizer Step callbacks.
    def on_forward_start(self, trainer, batch, **kwargs):
        pass

    def on_forward_end(self, trainer, loss, **kwargs):
        pass

    def on_backward_start(self, trainer, loss, **kwargs):
        pass

    def on_backward_end(self, trainer, **kwargs):
        pass

    def on_optimizer_step_start(self, trainer, **kwargs):
        pass

    def on_optimizer_step_end(self, trainer, **kwargs):
        pass

    # Error callback. Called when training fails with an exception.
    def on_error(self, trainer, exception, **kwargs):
        pass
