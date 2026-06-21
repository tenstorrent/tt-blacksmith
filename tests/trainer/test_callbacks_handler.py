# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from blacksmith.tools.trainer.callback import Callback
from blacksmith.tools.trainer.callbacks_handler import CallbackHandler
from blacksmith.tools.trainer.utils import normalize_callbacks

# TODO(mmilosevicTT): Add tests to CI once we have trainings through trainer class. See https://github.com/tenstorrent/tt-blacksmith/issues/629.


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

    handler("on_train_batch_start", batch=None)
    handler("on_train_batch_end")

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

    handler("on_train_start")

    assert received_trainers == [trainer]


def test_callback_handler_skips_missing_hooks():
    events = []

    class PartialCallback:
        def on_train_start(self, trainer):
            events.append("called")

    trainer = object()
    handler = CallbackHandler(trainer, [PartialCallback()])

    handler("on_train_batch_start", batch=None)

    assert events == []
