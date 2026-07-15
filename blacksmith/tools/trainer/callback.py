# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC


class Callback(ABC):
    # Training callbacks.
    def on_train_start(self, trainer, *args, **kwargs):
        pass

    def on_train_end(self, trainer, *args, **kwargs):
        pass

    def on_train_epoch_start(self, trainer, *args, **kwargs):
        pass

    def on_train_epoch_end(self, trainer, *args, **kwargs):
        pass

    def on_train_batch_start(self, trainer, batch, *args, **kwargs):
        pass

    def on_train_batch_end(self, trainer, *args, **kwargs):
        pass

    # Validation callbacks.
    def on_validation_start(self, trainer, *args, **kwargs):
        pass

    def on_validation_batch_start(self, trainer, batch, *args, **kwargs):
        pass

    def on_validation_batch_end(self, trainer, batch, loss, *args, **kwargs):
        pass

    def on_validation_end(self, trainer, *args, **kwargs):
        pass

    # Forward / Backward / Optimizer Step callbacks.
    def on_forward_start(self, trainer, batch, *args, **kwargs):
        pass

    def on_forward_end(self, trainer, loss, *args, **kwargs):
        pass

    def on_backward_start(self, trainer, loss, *args, **kwargs):
        pass

    def on_backward_end(self, trainer, *args, **kwargs):
        pass

    def on_optimizer_step_start(self, trainer, *args, **kwargs):
        pass

    def on_optimizer_step_end(self, trainer, *args, **kwargs):
        pass

    # Error callback.
    def on_error(self, trainer, exception, *args, **kwargs):
        pass
