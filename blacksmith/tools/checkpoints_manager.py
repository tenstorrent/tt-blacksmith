# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import torch

from blacksmith.experiments.torch.qwen.configs import TrainingConfig
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.storage_backends import StorageBackend


class CheckpointManager:
    def __init__(self, config: TrainingConfig, logger: TrainingLogger):
        self.config = config
        self.logger = logger

        self.checkpoint_dir = os.path.join(self.config.project_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.storage_backend = self._setup_storage_backend()

        self.checkpoint_history = self._load_checkpoint_history()

    def _setup_storage_backend(self) -> StorageBackend:
        """Setup storage backend based on config"""
        if self.config.storage_backend == "local":
            return None
        else:
            raise ValueError(f"Unknown storage backend: {self.config.storage_backend}")

    def _load_checkpoint_history(self) -> Dict[str, Any]:
        """Load checkpoint history from metadata file"""
        history_file = os.path.join(self.checkpoint_dir, "checkpoint_history.json")
        if os.path.exists(history_file):
            with open(history_file, "r") as f:
                return json.load(f)

        return {"checkpoints": [], "best_checkpoints": []}

    def _save_checkpoint_history(self):
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
        model: torch.nn.Module,
        step: int = 0,
        epoch: int = 0,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """
        Save a checkpoint

        Args:
            model: PyTorch model to save
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
            checkpoint_name = f"checkpoint_step{step}_epoch{epoch}_{timestamp}.pt"

        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        checkpoint_data = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        if self.config.save_optim and optimizer is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint_data, checkpoint_path)

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

    def _update_best_checkpoints(self, checkpoint_info: Dict[str, Any]):
        """Update list of best checkpoints based on metric"""
        metric_value = checkpoint_info["metrics"][self.config.checkpoint_metric]

        best_checkpoints = self.checkpoint_history.get("best_checkpoints", [])
        best_checkpoints.append({**checkpoint_info, "metric_value": metric_value})

        reverse = self.config.checkpoint_metric_mode == "max"
        best_checkpoints.sort(key=lambda x: x["metric_value"], reverse=reverse)

        self.checkpoint_history["best_checkpoints"] = best_checkpoints[: self.config.keep_best_n]

    def _cleanup_checkpoints(self):
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

    def load_checkpoint(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load checkpoint based on resume option in config"""
        if self.config.resume_option == "last":
            return self.load_latest_checkpoint(model, optimizer)
        elif self.config.resume_option == "best":
            return self.load_best_checkpoint(model, optimizer)
        elif self.config.resume_option == "path":
            if not self.config.checkpoint_path:
                raise ValueError("checkpoint_path must be provided when resume_option is 'path'")
            return self.load_checkpoint_path(self.config.checkpoint_path, model, optimizer)
        else:
            raise ValueError(f"Unknown resume_option: {self.config.resume_option}")

    def load_checkpoint_path(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
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
        if self.config.load_from_storage:
            self.storage_backend.load(checkpoint_path, checkpoint_path)

        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.logger.info("Loaded optimizer state")

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

    def load_latest_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
        """
        if not self.checkpoint_history["checkpoints"]:
            return None

        latest_checkpoint = self.checkpoint_history["checkpoints"][-1]
        return self.load_checkpoint_path(latest_checkpoint["path"], model, optimizer)

    def load_best_checkpoint(
        self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint based on tracked metric

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
        """
        if not self.checkpoint_history.get("best_checkpoints"):
            self.logger.warning("No best checkpoints found")
            return None

        best_checkpoint = self.checkpoint_history["best_checkpoints"][0]
        return self.load_checkpoint_path(best_checkpoint["path"], model, optimizer)

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about all checkpoints"""
        return {
            "total_checkpoints": len(self.checkpoint_history["checkpoints"]),
            "best_checkpoints": self.checkpoint_history.get("best_checkpoints", []),
            "latest_checkpoint": (
                self.checkpoint_history["checkpoints"][-1] if self.checkpoint_history["checkpoints"] else None
            ),
        }
