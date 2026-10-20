# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

import torch
from torch.distributed.tensor import DTensor, distribute_tensor

from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.crank.torch_helpers import to_host


class CrankCheckpointManager(CheckpointManager):
    """Checkpoint manager that (de)serializes torch model / optimizer state."""

    @staticmethod
    def align_state_dict_parameter_names(loaded_state_dict, model_state_dict):
        """
        Adjusts loaded_state_dict parameter names to match model_state_dict by adding
        or removing a prefix.

        This is necessary because "torch.compile" adds an "_orig_mod." prefix to
        the parameter names and depending on the case, this function either adds
        or removes the prefix.

        Logic:
        - Both have prefix or both don't: No change.
        - Model has it, Loaded doesn't: Prepend prefix to Loaded.
        - Loaded has it, Model doesn't: Strip prefix from Loaded.
        """
        prefix = "_orig_mod."

        loaded_keys = list(loaded_state_dict.keys())
        model_keys = list(model_state_dict.keys())

        if not loaded_keys or not model_keys:
            return loaded_state_dict

        loaded_has_prefix = loaded_keys[0].startswith(prefix)
        model_has_prefix = model_keys[0].startswith(prefix)

        if loaded_has_prefix == model_has_prefix:
            return loaded_state_dict

        new_state_dict = dict()

        if model_has_prefix and not loaded_has_prefix:
            for k, v in loaded_state_dict.items():
                new_state_dict[prefix + k] = v

        elif loaded_has_prefix and not model_has_prefix:
            for k, v in loaded_state_dict.items():
                new_state_dict[k.replace(prefix, "")] = v

        return new_state_dict

    @staticmethod
    def _trainable_names(model: torch.nn.Module) -> set[str]:
        return {name for name, param in model.named_parameters() if param.requires_grad}

    @staticmethod
    def _like(value: Any, reference: torch.Tensor) -> Any:
        """Lay out a host ``value`` like ``reference``: same mesh and placements if it is a DTensor."""
        if not isinstance(reference, DTensor) or not isinstance(value, torch.Tensor) or isinstance(value, DTensor):
            return value
        return distribute_tensor(value, reference.device_mesh, reference.placements)

    @classmethod
    def _distribute_optimizer_state(cls, optimizer: torch.optim.Optimizer) -> None:
        """Re-shard loaded optimizer state (e.g. Adam moments) to match its DTensor parameters.

        Scalar state such as ``step`` stays a plain tensor.
        """
        for param, state in optimizer.state.items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.shape == param.shape and value.dim() > 0:
                    state[key] = cls._like(value, param)

    def _write_state(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        metadata: Dict[str, Any],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        # Copy state to host (gathers DTensor shards); `tt` tensors cannot be pickled in place.
        trainable_names = self._trainable_names(model)
        state_dict = to_host({name: v for name, v in model.state_dict().items() if name in trainable_names})

        checkpoint_data = {
            "step": metadata["step"],
            "epoch": metadata["epoch"],
            "model_state_dict": state_dict,
            "metrics": metadata["metrics"],
            "timestamp": metadata["timestamp"],
        }

        if self.config.save_optim and optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = to_host(optimizer.state_dict())

        torch.save(checkpoint_data, checkpoint_path)

    def _read_state(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model_state_dict = model.state_dict()
        loaded_state_dict = self.align_state_dict_parameter_names(checkpoint["model_state_dict"], model_state_dict)
        loaded_state_dict = {
            k: self._like(v, model_state_dict[k]) if k in model_state_dict else v for k, v in loaded_state_dict.items()
        }
        # Only trainable parameters are saved: frozen ones may be missing, trainable ones must all be present.
        missing = sorted(self._trainable_names(model) - loaded_state_dict.keys())
        assert not missing, f"Checkpoint is missing trainable parameters: {missing}"
        result = model.load_state_dict(loaded_state_dict, strict=False)
        assert not result.unexpected_keys, f"Checkpoint has keys not in the model: {result.unexpected_keys}"

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._distribute_optimizer_state(optimizer)
            self.logger.info("Loaded optimizer state")

        return {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }
