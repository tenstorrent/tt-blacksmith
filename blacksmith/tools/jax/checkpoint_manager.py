# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from datetime import datetime
from functools import partial
from typing import Optional

import jax
from flax.serialization import from_bytes, from_state_dict, msgpack_serialize

from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.storage_backends import StorageBackend
from blacksmith.tools.templates.configs import TrainingConfig

logger = logging.getLogger(__name__)


class JaxCheckpointManager:
    """JAX/EasyDel counterpart to CheckpointManager, sharing its API and on-disk layout.
    Pytrees are msgpack-serialised on CPU so checkpoints are device-agnostic.
    """

    def __init__(
        self,
        config: TrainingConfig,
        logger: TrainingLogger,
    ) -> None:
        self.config = config
        self.logger = logger

        self.checkpoint_dir = os.path.join(self.config.project_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.storage_backend = self._setup_storage_backend()

        self.checkpoint_history = self._load_checkpoint_history()

    def _setup_storage_backend(self) -> Optional[StorageBackend]:
        """Setup storage backend based on config"""
        if self.config.storage_backend == "local":
            return None
        else:
            raise ValueError(f"Unknown storage backend: {self.config.storage_backend}")

    def _load_checkpoint_history(self) -> dict:
        """Load checkpoint history from metadata file"""
        history_file = os.path.join(self.checkpoint_dir, "checkpoint_history.json")
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                return json.load(f)

        return {"checkpoints": [], "best_checkpoints": []}

    def _save_checkpoint_history(self) -> None:
        """Save checkpoint history to metadata file"""
        history_file = os.path.join(self.checkpoint_dir, "checkpoint_history.json")
        with open(history_file, "w") as f:
            json.dump(self.checkpoint_history, f, indent=2)

    def should_save_checkpoint(self, step: int, epoch: Optional[int] = None) -> bool:
        """Determine if checkpoint should be saved at current step/epoch"""
        if epoch is not None:
            if self.config.save_strategy == "epoch":
                return epoch % self.config.epoch_freq == 0
            return False

        if self.config.save_strategy == "step":
            return step % self.config.steps_freq == 0
        return False

    def save_checkpoint(
        self,
        params,
        step: int = 0,
        epoch: int = 0,
        opt_state=None,
        rng=None,
        metrics: Optional[dict] = None,
        extra: Optional[dict] = None,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """Serialise a training snapshot to msgpack and update bookkeeping.

        Args:
            params: Trainable parameter pytree to save (e.g. nnx.LoRAParam state)
            step: Current training step
            epoch: Current epoch
            opt_state: Optimizer state (optional)
            rng: JAX PRNG key (optional)
            metrics: Dictionary of metrics (loss, accuracy, etc.)
            extra: Arbitrary JSON-serialisable extra payload
            checkpoint_name: Custom checkpoint name (auto-generated if None)

        Returns:
            Path to saved checkpoint
        """
        metrics = metrics or {}

        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_step{step}_epoch{epoch}_{timestamp}.msgpack"

        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        cpu_put = partial(jax.device_put, device=jax.devices("cpu")[0])

        checkpoint_data: dict = {
            "step": step,
            "epoch": epoch,
            "params": jax.tree.map(cpu_put, params),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        if self.config.save_optim and opt_state is not None:
            checkpoint_data["opt_state"] = jax.tree.map(cpu_put, opt_state)

        if rng is not None:
            checkpoint_data["rng"] = jax.tree.map(cpu_put, rng)

        if extra is not None:
            checkpoint_data["extra"] = extra

        with open(checkpoint_path, "wb") as f:
            f.write(msgpack_serialize(checkpoint_data))

        checkpoint_info = {
            "path": checkpoint_path,
            "name": checkpoint_name,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": checkpoint_data["timestamp"],
        }
        self.checkpoint_history["checkpoints"].append(checkpoint_info)

        if self.config.checkpoint_metric in metrics:
            self._update_best_checkpoints(checkpoint_info)
        self._cleanup_checkpoints()
        self._save_checkpoint_history()

        if self.config.sync_to_storage and self.config.remote_path:
            self.storage_backend.save(checkpoint_path)

        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        return checkpoint_path

    def _update_best_checkpoints(self, checkpoint_info: dict) -> None:
        """Update list of best checkpoints based on metric"""
        metric_value = checkpoint_info["metrics"][self.config.checkpoint_metric]

        best_checkpoints = self.checkpoint_history.get("best_checkpoints", [])
        best_checkpoints.append({**checkpoint_info, "metric_value": metric_value})

        reverse = self.config.checkpoint_metric_mode == "max"
        best_checkpoints.sort(key=lambda x: x["metric_value"], reverse=reverse)

        self.checkpoint_history["best_checkpoints"] = best_checkpoints[: self.config.keep_best_n]

    def _cleanup_checkpoints(self) -> None:
        """Keep only the last N and best N checkpoints"""
        all_checkpoints = self.checkpoint_history["checkpoints"]
        best_checkpoint_paths = {cp["path"] for cp in self.checkpoint_history.get("best_checkpoints", [])}

        if len(all_checkpoints) <= self.config.keep_last_n:
            return

        checkpoints_to_remove = all_checkpoints[: -self.config.keep_last_n]
        for checkpoint_info in checkpoints_to_remove:
            checkpoint_path = checkpoint_info["path"]

            # Don't remove if it's a best checkpoint
            if checkpoint_path not in best_checkpoint_paths:
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    self.logger.info(f"Removed old checkpoint: {checkpoint_path}")

        self.checkpoint_history["checkpoints"] = all_checkpoints[-self.config.keep_last_n :]

    def load_checkpoint(
        self,
        params_target,
        opt_state_target=None,
        rng_target=None,
    ) -> Optional[dict]:
        """Load checkpoint based on resume option in config"""
        if self.config.resume_option == "last":
            return self.load_latest_checkpoint(params_target, opt_state_target, rng_target)
        elif self.config.resume_option == "best":
            return self.load_best_checkpoint(params_target, opt_state_target, rng_target)
        elif self.config.resume_option == "path":
            if not self.config.checkpoint_path:
                raise ValueError("checkpoint_path must be provided when resume_option is 'path'")
            return self.load_checkpoint_path(self.config.checkpoint_path, params_target, opt_state_target, rng_target)
        else:
            raise ValueError(f"Unknown resume_option: {self.config.resume_option}")

    def load_checkpoint_path(
        self,
        checkpoint_path: str,
        params_target,
        opt_state_target=None,
        rng_target=None,
    ) -> dict:
        """Load a checkpoint, restoring pytrees into the supplied targets.
        Pass None for opt_state_target / rng_target to skip loading them.
        """
        if self.config.load_from_storage:
            self.storage_backend.load(checkpoint_path, checkpoint_path)

        with open(checkpoint_path, "rb") as f:
            raw = from_bytes(None, f.read())

        checkpoint = {
            "step": raw.get("step", 0),
            "epoch": raw.get("epoch", 0),
            "metrics": raw.get("metrics", {}),
            "params": from_state_dict(params_target, raw["params"]),
        }
        if "extra" in raw:
            checkpoint["extra"] = raw["extra"]

        if opt_state_target is not None and "opt_state" in raw:
            checkpoint["opt_state"] = from_state_dict(opt_state_target, raw["opt_state"])
            self.logger.info("Loaded optimizer state")

        if rng_target is not None and "rng" in raw:
            checkpoint["rng"] = from_state_dict(rng_target, raw["rng"])

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return checkpoint

    def load_latest_checkpoint(
        self,
        params_target,
        opt_state_target=None,
        rng_target=None,
    ) -> Optional[dict]:
        """
        Args:
            params_target: Initialised params pytree used as the deserialisation
                template
            opt_state_target: Optional optimizer-state template
            rng_target: Optional PRNG-key template
        """
        if not self.checkpoint_history["checkpoints"]:
            return None

        latest_checkpoint = self.checkpoint_history["checkpoints"][-1]
        return self.load_checkpoint_path(latest_checkpoint["path"], params_target, opt_state_target, rng_target)

    def load_best_checkpoint(
        self,
        params_target,
        opt_state_target=None,
        rng_target=None,
    ) -> Optional[dict]:
        """Load the best checkpoint based on tracked metric.

        Args:
            params_target: Initialised params pytree used as the deserialisation
                template
            opt_state_target: Optional optimizer-state template
            rng_target: Optional PRNG-key template
        """
        if not self.checkpoint_history.get("best_checkpoints"):
            self.logger.warning("No best checkpoints found")
            return None

        best_checkpoint = self.checkpoint_history["best_checkpoints"][0]
        return self.load_checkpoint_path(best_checkpoint["path"], params_target, opt_state_target, rng_target)

    def get_checkpoint_info(self) -> dict:
        """Get information about all checkpoints"""
        return {
            "total_checkpoints": len(self.checkpoint_history["checkpoints"]),
            "best_checkpoints": self.checkpoint_history.get("best_checkpoints", []),
            "latest_checkpoint": (
                self.checkpoint_history["checkpoints"][-1] if self.checkpoint_history["checkpoints"] else None
            ),
        }
