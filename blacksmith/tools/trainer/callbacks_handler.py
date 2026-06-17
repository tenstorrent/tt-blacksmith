# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence

from blacksmith.tools.trainer.callback import Callback


class CallbackHandler:
    def __init__(self, callbacks: Sequence[Callback]):
        self.callbacks = list(callbacks)

    def call(self, method_name: str, *args, **kwargs) -> None:
        if method_name.endswith("_start"):
            ordered_callbacks = self.callbacks
        elif method_name.endswith("_end"):
            ordered_callbacks = reversed(self.callbacks)
        else:
            raise ValueError(f"Callback hook {method_name!r} must end with '_start' or '_end'")

        for callback in ordered_callbacks:
            getattr(callback, method_name)(*args, **kwargs)
