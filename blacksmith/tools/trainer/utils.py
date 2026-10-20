# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from typing import Union

from blacksmith.tools.trainer.callback import Callback


def normalize_callbacks(
    callbacks: Union[Callback, Sequence[Callback], None],
) -> list[Callback]:
    if callbacks is None:
        return []
    if isinstance(callbacks, Callback):
        return [callbacks]
    return list(callbacks)
