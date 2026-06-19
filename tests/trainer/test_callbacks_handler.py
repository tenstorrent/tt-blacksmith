# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from blacksmith.tools.trainer.callback import Callback, CallbackEvent
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


def test_callback_handler_order():
    events = []
    trainer = object()
    handler = CallbackHandler(
        trainer,
        [
            RecordingCallback("first", events),
            RecordingCallback("second", events),
        ],
    )

    handler(CallbackEvent.ON_TRAIN_BATCH_START, batch=None)
    handler(CallbackEvent.ON_TRAIN_BATCH_END)

    assert events == [
        "first:on_train_batch_start",
        "second:on_train_batch_start",
        "first:on_train_batch_end",
        "second:on_train_batch_end",
    ]


def test_callback_handler_injects_trainer():
    received_trainers = []

    class TrainerCapturingCallback(Callback):
        def on_train_start(self, trainer):
            received_trainers.append(trainer)

    trainer = object()
    handler = CallbackHandler(trainer, [TrainerCapturingCallback()])

    handler(CallbackEvent.ON_TRAIN_START)

    assert received_trainers == [trainer]


def test_callback_event_values_match_callback_hooks():
    for event in CallbackEvent:
        assert hasattr(Callback, event)
