# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The contract every ttml model provider implements."""
from dataclasses import dataclass
from typing import Any, Dict, Protocol

import numpy as np

from blacksmith.configs import TrainerConfig


class UnsupportedTrainingConfig(ValueError):
    """This model cannot be trained with this config. The message names the setting and why."""


@dataclass
class AdapterState:
    """A LoRA adapter in PEFT form: PEFT tensor names -> arrays, and the ``adapter_config.json`` dict."""

    tensors: Dict[str, np.ndarray]
    config: Dict[str, Any]


class TtmlModelProvider(Protocol):
    def check(self, config: TrainerConfig, hf_config: Dict[str, Any]) -> None:
        """Raise ``UnsupportedTrainingConfig`` if ``config`` can't be trained on this model.

        Runs before any weights are downloaded or any device memory is allocated.
        """
        ...

    def build(self, config: TrainerConfig, hf_config: Dict[str, Any]) -> Any:
        """Load the pretrained model with LoRA adapters injected (a ``ttml.modules.LoraModel``).

        The returned model must take the standard ttml causal-LM input, which the trainer
        assumes for every model (collate, device upload and forward depend on it):
        ``logits = model(input_ids, None)`` with ``input_ids`` ``[B, 1, 1, T]`` uint32 and
        ``logits`` ``[B, 1, T, V]``, where ``None`` means a causal mask. A model that deviates
        is adapted here, or rejected by ``check``.
        """
        ...

    def export_adapter(self, model: Any, config: TrainerConfig) -> AdapterState:
        """The LoRA adapters of ``model`` (built from ``config``) as a PEFT adapter of the HF model."""
        ...

    def import_adapter(self, model: Any, state: AdapterState, config: TrainerConfig) -> None:
        """Load a PEFT adapter into ``model``; raise ``UnsupportedTrainingConfig`` if it doesn't fit."""
        ...
