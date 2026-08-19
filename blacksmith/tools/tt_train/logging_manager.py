# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Optional

TEST_LOGS_DIR = Path("tests/test_logs")


def _import_wandb():
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "wandb is required when use_wandb=True. Install it into the tt-metal python_env, "
            "or set use_wandb: false in the experiment YAML."
        ) from e
    return wandb


class TrainingLogger:
    def __init__(self, config: Any, test_log_filename_prefix: Optional[str] = None) -> None:
        self.config = config
        self.test_log_filename_prefix = test_log_filename_prefix
        self._wandb = None

        self._setup_std_logger()

        if self.config.use_wandb:
            self._setup_wandb()

        if self.test_log_filename_prefix is not None:
            self.train_log: list[dict[str, Any]] = []
            self.val_log: list[dict[str, Any]] = []
            TEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)
            self.csv_path_train = TEST_LOGS_DIR / f"{self.test_log_filename_prefix}_train.csv"
            self.csv_path_val = TEST_LOGS_DIR / f"{self.test_log_filename_prefix}_val.csv"

    def _setup_std_logger(self) -> None:
        self.std_logger = logging.getLogger(self.config.wandb_run_name)
        self.std_logger.setLevel(getattr(logging, self.config.log_level.upper()))
        self.std_logger.handlers.clear()

        formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
        console_handler.setFormatter(formatter)
        self.std_logger.addHandler(console_handler)

    def _setup_wandb(self) -> None:
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

    def info(self, message: str) -> None:
        self.std_logger.info(message)

    def warning(self, message: str) -> None:
        self.std_logger.warning(message)

    def error(self, message: str) -> None:
        self.std_logger.error(message)

    def debug(self, message: str) -> None:
        self.std_logger.debug(message)

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        metrics_str = " | ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
        step_str = f"Step {step} | " if step is not None else ""
        self.std_logger.info(f"{step_str}{metrics_str}")

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

    def log_summary(self, summary: dict[str, Any]) -> None:
        self.std_logger.info("Training Summary:")
        for key, value in summary.items():
            self.std_logger.info(f"  {key}: {value}")

        if self.config.use_wandb:
            try:
                for key, value in summary.items():
                    self.wandb_run.summary[key] = value
            except Exception as e:
                self.std_logger.warning(f"Failed to log summary to W&B: {e}")

    def _write_csv(self, path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def finish(self) -> None:
        if self.config.use_wandb:
            try:
                self._wandb.finish()
                self.std_logger.info("W&B run finished")
            except Exception as e:
                self.std_logger.warning(f"Failed to finish W&B run: {e}")

        if self.test_log_filename_prefix is not None:
            self._write_csv(self.csv_path_train, self.train_log, ["_step", "train/loss"])
            self._write_csv(self.csv_path_val, self.val_log, ["_step", "val/loss"])
            self.std_logger.info(f"Training and validation logs saved to {self.csv_path_train} and {self.csv_path_val}")
