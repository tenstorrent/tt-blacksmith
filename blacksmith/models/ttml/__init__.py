# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttml model providers: the only place that decides whether a model can be trained with a config.

The trainer, its callbacks and a job host never inspect LoRA targets, dtypes or sequence lengths.
They call ``get_model``, which either returns a trainable model or raises
``UnsupportedTrainingConfig`` with the reason. Bringing up a new model means adding one provider
to ``PROVIDERS``; nothing else changes.

Importing this package does not import ttml, so configs can be checked on a host without it.
"""
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from blacksmith.configs import TrainerConfig
from blacksmith.models.ttml.llama import LlamaProvider
from blacksmith.models.ttml.provider import (
    AdapterState,
    TtmlModelProvider,
    UnsupportedTrainingConfig,
)

__all__ = [
    "AdapterState",
    "PROVIDERS",
    "TtmlModelProvider",
    "UnsupportedTrainingConfig",
    "get_model",
    "get_provider",
    "load_hf_config",
]

# Keyed by `model_type` in the HF config.json.
PROVIDERS: Dict[str, TtmlModelProvider] = {"llama": LlamaProvider()}


def load_hf_config(model_name: str) -> Dict[str, Any]:
    """The raw HF ``config.json`` of a local model directory or a hub model (a small download)."""
    path = Path(model_name) / "config.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(model_name, "config.json"))
    return json.loads(path.read_text())


def get_provider(config: TrainerConfig) -> Tuple[TtmlModelProvider, Dict[str, Any]]:
    hf_config = load_hf_config(config.model_name)
    model_type = hf_config.get("model_type")
    provider = PROVIDERS.get(model_type)
    if provider is None:
        raise UnsupportedTrainingConfig(
            f"model_name: no ttml provider for model_type {model_type!r} ({config.model_name}); "
            f"available: {sorted(PROVIDERS)}"
        )
    return provider, hf_config


def get_model(config: TrainerConfig) -> Any:
    """Build the trainable ttml model for ``config``, or raise ``UnsupportedTrainingConfig``."""
    provider, hf_config = get_provider(config)
    # Fail fast: before the weight download and before any device memory is allocated.
    provider.check(config, hf_config)
    return provider.build(config, hf_config)
