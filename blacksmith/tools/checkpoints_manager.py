# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from blacksmith.configs import CheckpointConfig
from blacksmith.tools.logging_manager import TrainingLogger


class CheckpointManager(ABC):
    """Framework-agnostic checkpoint bookkeeping.

    The base owns the on-disk history (metadata file, best / last-N rotation,
    init dispatch) and must stay importable without a deep-learning framework.
    Reading and writing the actual model / optimizer state is delegated to the
    abstract ``_write_state`` / ``_read_state`` hooks, implemented by a concrete
    subclass such as ``CrankCheckpointManager``.
    """

    # Appended to auto-generated checkpoint names. Empty for frameworks whose checkpoints are directories.
    checkpoint_suffix = ".pt"

    def __init__(
        self,
        config: CheckpointConfig,
        logger: Optional[TrainingLogger] = None,
    ):
        self.config = config
        # Fall back to a plain module logger when no TrainingLogger is provided
        # (e.g. when used standalone via the checkpoint callback). Both expose
        # the .info/.warning methods used below.
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.checkpoint_dir = os.path.join(self.config.project_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.history_file = os.path.join(self.checkpoint_dir, "checkpoint_history.json")

        self.checkpoint_history = self._start_checkpoint_history()

    def _start_checkpoint_history(self) -> Dict[str, Any]:
        """Start an empty history, also when this run is initialized from a checkpoint.

        A run must not inherit an earlier run's entries from the same ``project_dir``: they would
        compete for best-N, get their files rotated away by this run, go stale when this run
        overwrites a fixed-name checkpoint such as the final one, and mix their step numbers with
        this run's, which always start from 0. The earlier history is only read to find the
        "last" / "best" checkpoint to initialize from.
        """
        if os.path.exists(self.history_file):
            self.logger.warning(
                f"Starting a new checkpoint history; the existing {self.history_file} is replaced on the first save"
            )

        return {"checkpoints": [], "best_checkpoints": []}

    def _read_saved_history(self) -> Dict[str, Any]:
        """Read the metadata file, which holds the previous run's history until this run first saves."""
        if not os.path.exists(self.history_file):
            return {"checkpoints": [], "best_checkpoints": []}
        with open(self.history_file, "r") as f:
            return json.load(f)

    def _save_checkpoint_history(self):
        """Save checkpoint history to metadata file"""
        with open(self.history_file, "w") as f:
            json.dump(self.checkpoint_history, f, indent=2)

    # ------------------------------------------------------- framework hooks
    @abstractmethod
    def _write_state(
        self,
        checkpoint_path: str,
        model: Any,
        metadata: Dict[str, Any],
        optimizer: Optional[Any] = None,
    ) -> None:
        """Serialize ``model`` (and optionally ``optimizer``) plus ``metadata`` to disk."""
        pass

    @abstractmethod
    def _read_state(
        self,
        checkpoint_path: str,
        model: Any,
        optimizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load state from disk into ``model`` (and optionally ``optimizer``).

        Returns the stored metadata (``step`` / ``epoch`` / ``metrics``).
        """
        pass

    def should_save_checkpoint(self, step: int, epoch: Optional[int] = None) -> bool:
        """Determine if checkpoint should be saved at current step/epoch"""
        if epoch is not None:
            if self.config.save_strategy == "epoch":
                return (epoch + 1) % self.config.epoch_freq == 0
            return False

        if self.config.save_strategy == "step":
            return step % self.config.steps_freq == 0
        return False

    def save_checkpoint(
        self,
        model: Any,
        step: int = 0,
        epoch: int = 0,
        optimizer: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """
        Save a checkpoint

        Args:
            model: Model to save
            step: Current training step
            epoch: Current epoch
            optimizer: Optimizer state (optional)
            metrics: Dictionary of metrics (loss, accuracy, etc.)
            checkpoint_name: Custom checkpoint name (auto-generated if None)

        Returns:
            Path to saved checkpoint
        """
        metrics = metrics or {}

        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_step{step}_epoch{epoch}_{timestamp}{self.checkpoint_suffix}"

        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        metadata = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        self._write_state(checkpoint_path, model, metadata, optimizer=optimizer)

        checkpoint_info = {
            "path": checkpoint_path,
            "name": checkpoint_name,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": metadata["timestamp"],
        }
        tracked_before = self._tracked_paths() | {checkpoint_path}
        self._drop_entries(checkpoint_path)
        self.checkpoint_history["checkpoints"].append(checkpoint_info)

        if self.config.checkpoint_metric in metrics:
            self._update_best_checkpoints(checkpoint_info)
        if not self.config.keep_all_checkpoints:
            self._rotate_last_checkpoints()
            self._remove_untracked(tracked_before)
        self._save_checkpoint_history()

        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        return checkpoint_path

    def _update_best_checkpoints(self, checkpoint_info: Dict[str, Any]):
        """Update list of best checkpoints based on metric"""
        metric_value = checkpoint_info["metrics"][self.config.checkpoint_metric]

        best_checkpoints = self.checkpoint_history.get("best_checkpoints", [])
        best_checkpoints.append({**checkpoint_info, "metric_value": metric_value})

        reverse = self.config.checkpoint_metric_mode == "max"
        best_checkpoints.sort(key=lambda x: x["metric_value"], reverse=reverse)

        self.checkpoint_history["best_checkpoints"] = best_checkpoints[: self.config.keep_best_n]

    def _drop_entries(self, checkpoint_path: str):
        """Remove every entry for ``checkpoint_path`` from both lists.

        Called after the file is overwritten, so the old entries (and their metrics) describe
        weights that are gone.
        """
        for key in ("checkpoints", "best_checkpoints"):
            self.checkpoint_history[key] = [
                cp for cp in self.checkpoint_history.get(key, []) if cp["path"] != checkpoint_path
            ]

    def _tracked_paths(self) -> set[str]:
        """Paths referenced by either the last-N or the best-N list."""
        return {
            cp["path"] for key in ("checkpoints", "best_checkpoints") for cp in self.checkpoint_history.get(key, [])
        }

    def _rotate_last_checkpoints(self):
        """Trim the last-N list. Files are removed separately by `_remove_untracked`."""
        self.checkpoint_history["checkpoints"] = self.checkpoint_history["checkpoints"][-self.config.keep_last_n :]

    def _remove_untracked(self, previously_tracked: set[str]):
        """Delete files that were tracked before this save but are in neither list now.

        A file is kept while either list references it, so a checkpoint that is
        protected by best-N after leaving last-N is still deleted once it also
        falls out of best-N.
        """
        for checkpoint_path in previously_tracked - self._tracked_paths():
            if os.path.isdir(checkpoint_path):
                shutil.rmtree(checkpoint_path)
                self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
            elif os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                self.logger.info(f"Removed old checkpoint: {checkpoint_path}")

    def load_checkpoint(self, model: Any, optimizer: Optional[Any] = None) -> Dict[str, Any]:
        """Load checkpoint based on init option in config"""
        if self.config.init_option == "last":
            return self.load_latest_checkpoint(model, optimizer)
        elif self.config.init_option == "best":
            return self.load_best_checkpoint(model, optimizer)
        elif self.config.init_option == "path":
            if not self.config.checkpoint_path:
                raise ValueError("checkpoint_path must be provided when init_option is 'path'")
            return self.load_checkpoint_path(self.config.checkpoint_path, model, optimizer)
        else:
            raise ValueError(f"Unknown init_option: {self.config.init_option}")

    def load_checkpoint_path(
        self,
        checkpoint_path: str,
        model: Any,
        optimizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)

        Returns:
            Dictionary containing checkpoint metadata
        """
        metadata = self._read_state(checkpoint_path, model, optimizer)
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return metadata

    def load_latest_checkpoint(
        self,
        model: Any,
        optimizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load the most recent checkpoint in the saved history

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)

        Raises:
            FileNotFoundError: If the saved history has no checkpoint
        """
        checkpoints = self._read_saved_history()["checkpoints"]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint recorded in {self.history_file}")

        return self.load_checkpoint_path(checkpoints[-1]["path"], model, optimizer)

    def load_best_checkpoint(self, model: Any, optimizer: Optional[Any] = None) -> Dict[str, Any]:
        """Load the best checkpoint in the saved history based on tracked metric

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)

        Raises:
            FileNotFoundError: If the saved history has no best checkpoint
        """
        best_checkpoints = self._read_saved_history().get("best_checkpoints")
        if not best_checkpoints:
            raise FileNotFoundError(f"No best checkpoint recorded in {self.history_file}")

        return self.load_checkpoint_path(best_checkpoints[0]["path"], model, optimizer)

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about all checkpoints"""
        return {
            "total_checkpoints": len(self.checkpoint_history["checkpoints"]),
            "best_checkpoints": self.checkpoint_history.get("best_checkpoints", []),
            "latest_checkpoint": (
                self.checkpoint_history["checkpoints"][-1] if self.checkpoint_history["checkpoints"] else None
            ),
        }
