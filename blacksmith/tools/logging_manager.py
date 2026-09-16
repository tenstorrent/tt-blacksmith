# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stdout + W&B training logger, with optional CSV capture for CI."""
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from blacksmith.tools.configs import TrainingConfig

TEST_LOGS_DIR = Path("tests/test_logs")


def _import_wandb():
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "wandb is required when use_wandb=True. Activate the project "
            "environment with `source env/activate --crank` so it gets installed."
        ) from e
    return wandb


class TrainingLogger:
    def __init__(self, config: TrainingConfig, test_log_filename_prefix: Optional[str] = None):
        self.config = config
        self.test_log_filename_prefix = test_log_filename_prefix
        self._wandb = None

        self.std_logger = logging.getLogger(config.wandb_run_name)
        self.std_logger.setLevel(getattr(logging, config.log_level.upper()))
        self.std_logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        self.std_logger.addHandler(handler)

        if config.use_wandb:
            self._setup_wandb()

        if test_log_filename_prefix is not None:
            self.train_log: list[dict] = []
            self.val_log: list[dict] = []
            TEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)
            self.csv_path_train = TEST_LOGS_DIR / f"{test_log_filename_prefix}_train.csv"
            self.csv_path_val = TEST_LOGS_DIR / f"{test_log_filename_prefix}_val.csv"

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
        self.std_logger.info(message)

    def warning(self, message: str):
        self.std_logger.warning(message)

    def error(self, message: str, traceback_str: Optional[str] = None):
        self.std_logger.error(message)
        if self.config.use_wandb:
            self.wandb_run.log({"error": message, "traceback": traceback_str})

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        if metrics:
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()
            )
            self.std_logger.info(f"{f'Step {step} | ' if step is not None else ''}{metrics_str}")

        if self.config.use_wandb:
            try:
                self.wandb_run.log(metrics, step=step, commit=commit)
            except Exception as e:
                self.std_logger.warning(f"Failed to log to W&B: {e}")

        if self.test_log_filename_prefix is not None:
            if "train/loss" in metrics:
                self.train_log.append({"_step": step, "train/loss": metrics["train/loss"]})
            if "val/loss" in metrics:
                self.val_log.append({"_step": step, "val/loss": metrics["val/loss"]})

    def finish(self):
        if self.config.use_wandb and self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception as e:
                self.std_logger.warning(f"Failed to finish W&B run: {e}")

        if self.test_log_filename_prefix is not None:
            import pandas as pd

            pd.DataFrame(self.train_log or [], columns=["_step", "train/loss"]).to_csv(
                self.csv_path_train, index=False, float_format="%.10g"
            )
            pd.DataFrame(self.val_log or [], columns=["_step", "val/loss"]).to_csv(
                self.csv_path_val, index=False, float_format="%.10g"
            )
            self.std_logger.info(f"Logs saved to {self.csv_path_train} and {self.csv_path_val}")
