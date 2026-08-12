# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from blacksmith.tools.configs import LoggingConfig

TEST_LOGS_DIR = Path("tests/test_logs")
GOLDEN_LOGS_DIR = Path("tests/golden_files")


def _import_wandb():
    """Import wandb only when W&B logging is enabled.

    Keeps wandb an optional dependency for stdout-only consumers
    (e.g. tt-media-server sets use_wandb=False).
    """
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "wandb is required when use_wandb=True. "
            "Install it with `pip install wandb`."
        ) from e
    return wandb


class TrainingLogger:
    def __init__(self, config: LoggingConfig, test_log_filename_prefix: Optional[str] = None):
        self.config = config
        self.test_log_filename_prefix = test_log_filename_prefix
        self._wandb = None

        self._setup_std_logger()

        if self.config.use_wandb:
            self._setup_wandb()

        if self.test_log_filename_prefix is not None:
            self.val_log = []
            self.train_log = []

            TEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)

            self.csv_path_train = TEST_LOGS_DIR / f"{self.test_log_filename_prefix}_train.csv"
            self.csv_path_val = TEST_LOGS_DIR / f"{self.test_log_filename_prefix}_val.csv"

    def _setup_std_logger(self):
        self.std_logger = logging.getLogger(self.config.wandb_run_name)
        self.std_logger.setLevel(getattr(logging, self.config.log_level.upper()))

        # Remove existing handlers to avoid duplicates
        self.std_logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # Console handler (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
        console_handler.setFormatter(formatter)
        self.std_logger.addHandler(console_handler)

    def _setup_wandb(self):
        self.std_logger.info("Initializing Weights & Biases (W&B)...")
        try:
            self._wandb = _import_wandb()
            self.wandb_run = self._wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                tags=self.config.wandb_tags,
                config=self.config.model_dump(),
                save_code=True,
            )
        except Exception as e:
            self.std_logger.error(f"Failed to initialize W&B: {e}")
            self.config.use_wandb = False
            self._wandb = None

    def info(self, message: str):
        """Log info message to stdout"""
        self.std_logger.info(message)

    def warning(self, message: str):
        """Log warning message to stdout"""
        self.std_logger.warning(message)

    def error(self, message: str, traceback_str: Optional[str] = None):
        """Log error message to stdout"""
        self.std_logger.error(message)

        if self.config.use_wandb:
            self.wandb_run.alert(
                title="Training Failed",
                text=message,
                level=self._wandb.AlertLevel.ERROR,
            )
            self.wandb_run.log({"error": message, "traceback": traceback_str})

    def debug(self, message: str):
        """Log debug message to stdout"""
        self.std_logger.debug(message)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to both stdout, CSV files (if test_config is set), and W&B.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step number
            commit: Whether to commit to W&B (batches logs if False)
        """
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        step_str = f"Step {step} | " if step is not None else ""
        self.std_logger.info(f"{step_str}{metrics_str}")

        if self.config.use_wandb:
            try:
                self.wandb_run.log(metrics, step=step, commit=commit)
            except Exception as e:
                self.std_logger.warning(f"Failed to log to W&B: {e}")

        if self.test_log_filename_prefix is not None:
            row = {"_step": step}
            if "train/loss" in metrics:
                row["train/loss"] = metrics["train/loss"]
            if "train/reward_mean" in metrics:
                row["train/reward_mean"] = metrics["train/reward_mean"]
            if len(row) > 1:
                self.train_log.append(row)
            if "val/loss" in metrics:
                self.val_log.append({"_step": step, "val/loss": metrics["val/loss"]})

    def log_image(self, key: str, image: Any, step: Optional[int] = None, caption: str = "", commit: bool = False):
        """Log a PIL image to W&B (no-op on stdout-only runs besides a log line)."""
        if self.config.use_wandb:
            try:
                image_payload = {key: self._wandb.Image(image, caption=caption)}
                self.wandb_run.log(image_payload, step=step, commit=commit)
            except Exception as e:
                self.std_logger.warning(f"Failed to log image to W&B: {e}")
        else:
            self.std_logger.info(f"[{key}] step={step} {caption}")

    def log_video(
        self, key: str, frames_uint8: "np.ndarray", fps: int, step: Optional[int] = None, commit: bool = False
    ):
        """Log a video to W&B."""
        if self.config.use_wandb:
            try:
                arr = np.transpose(frames_uint8, (0, 3, 1, 2))
                video_payload = {key: self._wandb.Video(arr, fps=fps, format="mp4")}
                self.wandb_run.log(video_payload, step=step, commit=commit)
            except Exception as e:
                self.std_logger.warning(f"Failed to log video to W&B: {e}")

    def log_model_info(self, model_info: Dict[str, Any]):
        """
        Log model information (architecture, parameters, etc.).

        Args:
            model_info: Dictionary of model information
        """
        self.std_logger.info("Model Information:")
        for key, value in model_info.items():
            self.std_logger.info(f"  {key}: {value}")

        if self.config.use_wandb:
            try:
                self.wandb_run.config.update({"model": model_info})
            except Exception as e:
                self.std_logger.warning(f"Failed to log model info to W&B: {e}")

    def watch_model(self, model: torch.nn.Module):
        """
        Watch model gradients and parameters in W&B.

        Args:
            model: PyTorch model to watch
        """
        if self.config.use_wandb and self.config.model_to_wandb:
            try:
                self.wandb_run.watch(model, log=self.config.wandb_watch_mode, log_freq=self.config.wandb_log_freq)
                self.std_logger.info("W&B model watching enabled")
            except Exception as e:
                self.std_logger.warning(f"Failed to watch model in W&B: {e}")

    def log_artifact(self, artifact_path: str, artifact_type: str = "model", name: Optional[str] = None):
        """
        Log an artifact (model, dataset, etc.) to W&B.

        Args:
            artifact_path: Path to artifact
            artifact_type: Type of artifact (model, dataset, etc.)
            name: Artifact name (defaults to filename)
        """
        if self.config.use_wandb:
            try:
                artifact_name = name or os.path.basename(artifact_path)
                artifact = self._wandb.Artifact(artifact_name, type=artifact_type)
                artifact.add_file(artifact_path)

                self.wandb_run.log_artifact(artifact)
                self.std_logger.info(f"Logged artifact '{artifact_name}' to W&B")
            except Exception as e:
                self.std_logger.warning(f"Failed to log artifact to W&B: {e}")

    def log_summary(self, summary: Dict[str, Any]):
        """
        Log final summary statistics.

        Args:
            summary: Dictionary of summary statistics
        """
        self.std_logger.info("Training Summary:")
        for key, value in summary.items():
            self.std_logger.info(f"  {key}: {value}")

        if self.config.use_wandb:
            try:
                for key, value in summary.items():
                    self.wandb_run.run.summary[key] = value
            except Exception as e:
                self.std_logger.warning(f"Failed to log summary to W&B: {e}")

    def finish(self):
        if self.config.use_wandb:
            try:
                self._wandb.finish()
                self.std_logger.info("W&B run finished")
            except Exception as e:
                self.std_logger.warning(f"Failed to finish W&B run: {e}")

        if self.test_log_filename_prefix is not None:
            train_df = pd.DataFrame(self.train_log) if self.train_log else pd.DataFrame(columns=["_step", "train/loss"])
            train_df.to_csv(self.csv_path_train, index=False, float_format="%.10g")
            # Skip val CSV when validation is disabled.
            if getattr(self.config, "do_validation", True):
                val_df = pd.DataFrame(self.val_log) if self.val_log else pd.DataFrame(columns=["_step", "val/loss"])
                val_df.to_csv(self.csv_path_val, index=False, float_format="%.10g")
                self.std_logger.info(
                    f"Training and validation logs saved to {self.csv_path_train} and {self.csv_path_val}"
                )
            else:
                self.std_logger.info(f"Training log saved to {self.csv_path_train}")
