# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Union

from blacksmith.tools.trainer.callback import Callback
from blacksmith.tools.trainer.callbacks_handler import CallbackHandler
from blacksmith.tools.trainer.utils import normalize_callbacks


class Trainer(ABC):
    def __init__(
        self,
        callbacks: Union[Callback, Sequence[Callback], None] = None,
    ):
        self.config = None
        self.callback_handler = CallbackHandler(self, normalize_callbacks(callbacks))

    @abstractmethod
    def setup(
        self,
        config: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Setup the trainer with the given configuration.
        """
        pass

    @abstractmethod
    def train(self) -> None:
        """
        Train the model on the training dataset.
        """
        pass

    @abstractmethod
    def validate(self) -> None:
        """
        Validate the model on the validation dataset.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up any resources used by the trainer.
        """
        pass
