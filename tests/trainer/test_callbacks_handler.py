# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from blacksmith.tools.trainer.callback import Callback
from blacksmith.tools.trainer.callbacks_handler import CallbackHandler
from blacksmith.tools.trainer.utils import normalize_callbacks


class RecordingCallback(Callback):
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events

    def on_train_batch_start(self, trainer, batch):
        self.events.append(f"{self.name}:on_train_batch_start")

    def on_train_batch_end(self, trainer):
        self.events.append(f"{self.name}:on_train_batch_end")


def test_normalize_callbacks():
    assert normalize_callbacks(None) == []
    callback = RecordingCallback("a", [])
    assert normalize_callbacks(callback) == [callback]
    assert normalize_callbacks([callback]) == [callback]


def test_callback_handler_stack_order():
    events = []
    handler = CallbackHandler(
        [
            RecordingCallback("first", events),
            RecordingCallback("second", events),
        ]
    )

    handler.call("on_train_batch_start", trainer=None, batch=None)
    handler.call("on_train_batch_end", trainer=None)

    assert events == [
        "first:on_train_batch_start",
        "second:on_train_batch_start",
        "second:on_train_batch_end",
        "first:on_train_batch_end",
    ]


def test_callback_handler_rejects_invalid_hook_name():
    handler = CallbackHandler([])
    try:
        handler.call("on_train_begin", trainer=None)
        assert False, "expected ValueError"
    except ValueError as error:
        assert "_start' or '_end'" in str(error)
