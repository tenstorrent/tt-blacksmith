# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

from blacksmith.configs import CheckpointConfig, TrainerConfig
from blacksmith.models.ttml.provider import TtmlModelProvider
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.ttml.utilities import read_adapter, write_adapter


class TtmlCheckpointManager(CheckpointManager):
    """Checkpoints are PEFT LoRA adapter directories.

    Generic: the model provider maps the model's adapters to and from PEFT tensors; this class only
    reads and writes the files. Only adapter weights are saved; there is no optimizer state, which
    only a resume would need.
    """

    # Checkpoints are directories.
    checkpoint_suffix = ""

    def __init__(
        self,
        config: CheckpointConfig,
        provider: TtmlModelProvider,
        run_config: TrainerConfig,
        logger: Optional[TrainingLogger] = None,
    ):
        if config.save_optim:
            raise NotImplementedError(
                "checkpoint.save_optim: the ttml trainer saves LoRA adapters only (no resume); set it to False"
            )
        super().__init__(config, logger)
        self.provider = provider
        self.run_config = run_config

    def _write_state(
        self,
        checkpoint_path: str,
        model: Any,
        metadata: Dict[str, Any],
        optimizer: Optional[Any] = None,
    ) -> None:
        write_adapter(checkpoint_path, self.provider.export_adapter(model, self.run_config), metadata)

    def _read_state(
        self,
        checkpoint_path: str,
        model: Any,
        optimizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        state, metadata = read_adapter(checkpoint_path)
        self.provider.import_adapter(model, state, self.run_config)
        return {
            "step": metadata.get("step", 0),
            "epoch": metadata.get("epoch", 0),
            "metrics": metadata.get("metrics", {}),
        }
