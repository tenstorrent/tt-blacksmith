# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import sys
from typing import Any, Dict, Optional

import torch
import wandb

from blacksmith.tools.templates.configs import TrainingConfig


class TrainingLogger:
    def __init__(self, config: TrainingConfig):
        self.config = config

        self._setup_std_logger()

        if self.config.use_wandb:
            self._setup_wandb()

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
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                tags=self.config.wandb_tags,
                config=self.config.model_dump(),
                save_code=True,
            )
        except Exception as e:
            self.std_logger.error(f"Failed to initialize W&B: {e}")
            self.config.use_wandb = False

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
            self.wandb_run.alert(title="Training Failed", text=message, level=wandb.AlertLevel.ERROR)
            self.wandb_run.log({"error": message, "traceback": traceback_str})

    def debug(self, message: str):
        """Log debug message to stdout"""
        self.std_logger.debug(message)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to both stdout and W&B.

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
                artifact = wandb.Artifact(artifact_name, type=artifact_type)
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
                wandb.finish()
                self.std_logger.info("W&B run finished")
            except Exception as e:
                self.std_logger.warning(f"Failed to finish W&B run: {e}")
