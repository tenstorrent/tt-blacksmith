# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Any

from blacksmith.tools.trainer.callback import Callback


class CallbackHandler:
    def __init__(self, trainer, callbacks: Sequence[Callback]):
        self.trainer = trainer
        self.callbacks = list(callbacks)

    def __call__(self, method_name: str, *args, **kwargs) -> None:
        for callback in self.callbacks:
            getattr(callback, method_name)(self.trainer, *args, **kwargs)
