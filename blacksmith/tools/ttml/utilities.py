# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PEFT adapter directories on disk. Pure numpy / safetensors: no ttml import.

Layout (what ``PeftModel.save_pretrained`` writes, plus blacksmith's metadata, which PEFT ignores)::

    <dir>/adapter_config.json
    <dir>/adapter_model.safetensors
    <dir>/trainer_state.json        # step, epoch, metrics, timestamp
"""
import json
import os
import shutil
from typing import Any, Dict, Optional, Tuple

import numpy as np

from blacksmith.models.ttml.provider import AdapterState

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
TRAINER_STATE = "trainer_state.json"


def write_adapter(path: str, state: AdapterState, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Write ``state`` as a PEFT adapter directory at ``path``, replacing one that is there.

    Written next to ``path`` first and moved into place, so a failed save never leaves a partial adapter.
    """
    from safetensors.numpy import save_file

    staging = path + ".tmp"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    with open(os.path.join(staging, ADAPTER_CONFIG), "w") as f:
        json.dump(state.config, f, indent=2)
    save_file(
        {name: np.ascontiguousarray(value, dtype=np.float32) for name, value in state.tensors.items()},
        os.path.join(staging, ADAPTER_WEIGHTS),
    )
    if metadata is not None:
        with open(os.path.join(staging, TRAINER_STATE), "w") as f:
            json.dump(metadata, f, indent=2)
    shutil.rmtree(path, ignore_errors=True)
    os.replace(staging, path)


def read_adapter(path: str) -> Tuple[AdapterState, Dict[str, Any]]:
    """Read a PEFT adapter directory: the adapter, and blacksmith's metadata (``{}`` if there is none)."""
    from safetensors.numpy import load_file

    if not os.path.isdir(path):
        raise FileNotFoundError(f"No PEFT adapter directory at {path}")
    with open(os.path.join(path, ADAPTER_CONFIG)) as f:
        config = json.load(f)
    tensors = load_file(os.path.join(path, ADAPTER_WEIGHTS))
    metadata = {}
    state_file = os.path.join(path, TRAINER_STATE)
    if os.path.exists(state_file):
        with open(state_file) as f:
            metadata = json.load(f)
    return AdapterState(tensors=tensors, config=config), metadata
