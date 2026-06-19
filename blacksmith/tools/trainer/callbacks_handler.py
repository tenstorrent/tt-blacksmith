# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from blacksmith.tools.trainer.callback import Callback, CallbackEvent


class CallbackHandler:
    def __init__(self, trainer, callbacks: Sequence[Callback]):
        self.trainer = trainer
        self.callbacks = list(callbacks)

    def __call__(self, event: CallbackEvent, *args, **kwargs) -> None:
        for callback in self.callbacks:
            getattr(callback, event)(self.trainer, *args, **kwargs)
