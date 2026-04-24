# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import jax

from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.templates.configs import TrainingConfig

logger = logging.getLogger(__name__)


class JaxCheckpointManager:
    """Manage JAX training checkpoints (pickle + JSON history).

    Args:
        config: Training configuration (inherits checkpoint
            fields from the base ``TrainingConfig``).
        training_logger: Shared :class:`TrainingLogger`.
    """

    def __init__(
        self,
        config: TrainingConfig,
        training_logger: TrainingLogger,
    ) -> None:
        self.config = config
        self.training_logger = training_logger

        self.checkpoint_dir = Path(self.config.project_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_history = self._load_checkpoint_history()

    def _history_path(self) -> Path:
        return self.checkpoint_dir / "checkpoint_history.json"

    def _load_checkpoint_history(self) -> dict:
        hp = self._history_path()
        if hp.exists():
            with open(hp) as f:
                return json.load(f)
        return {"checkpoints": [], "best_checkpoints": []}

    def _save_checkpoint_history(self) -> None:
        with open(self._history_path(), "w") as f:
            json.dump(self.checkpoint_history, f, indent=2)

    def should_save_checkpoint(
        self,
        step: int,
        epoch: Optional[int] = None,
    ) -> bool:
        """Decide whether to checkpoint at *step* / *epoch*."""
        if epoch is not None:
            if self.config.save_strategy == "epoch":
                return epoch % self.config.epoch_freq == 0
            return False
        if self.config.save_strategy == "step":
            return step % self.config.steps_freq == 0
        return False

    def save_checkpoint(
        self,
        *,
        step: int,
        epoch: int,
        params,
        opt_state=None,
        rng=None,
        metrics: Optional[dict] = None,
        extra: Optional[dict] = None,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """Persist a checkpoint to disk (pickle).

        All pytrees are moved to CPU before serialisation so
        that checkpoints are device-agnostic.

        Args:
            step: Current global training step.
            epoch: Current epoch number.
            params: Model parameters (pytree).
            opt_state: Optimizer state (optional).
            rng: JAX PRNG key (optional).
            metrics: Scalar metrics dict (optional).
            extra: Arbitrary extra payload (optional).
            checkpoint_name: Custom filename (auto-generated
                when *None*).

        Returns:
            Absolute path to the saved checkpoint file.
        """
        metrics = metrics or {}

        if checkpoint_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"checkpoint_step{step}_epoch{epoch}_{ts}.pkl"

        path = self.checkpoint_dir / checkpoint_name

        cpu = jax.devices("cpu")[0]
        cpu_put = lambda t: jax.tree.map(lambda x: jax.device_put(x, cpu), t)

        data: dict = {
            "step": step,
            "epoch": epoch,
            "params": cpu_put(params),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        if self.config.save_optim and opt_state is not None:
            data["opt_state"] = cpu_put(opt_state)

        if rng is not None:
            data["rng"] = cpu_put(rng)

        if extra is not None:
            data["extra"] = extra

        with open(path, "wb") as f:
            pickle.dump(data, f)

        info: dict = {
            "path": str(path),
            "name": checkpoint_name,
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": data["timestamp"],
        }
        self.checkpoint_history["checkpoints"].append(info)

        if self.config.checkpoint_metric in metrics:
            self._update_best_checkpoints(info)
        self._cleanup_checkpoints()
        self._save_checkpoint_history()

        self.training_logger.info(f"Saved checkpoint: {path}")
        return str(path)

    def _update_best_checkpoints(self, info: dict) -> None:
        metric_value = info["metrics"][self.config.checkpoint_metric]
        best = self.checkpoint_history.get("best_checkpoints", [])
        best.append({**info, "metric_value": metric_value})

        reverse = self.config.checkpoint_metric_mode == "max"
        best.sort(key=lambda x: x["metric_value"], reverse=reverse)
        self.checkpoint_history["best_checkpoints"] = best[: self.config.keep_best_n]

    def _cleanup_checkpoints(self) -> None:
        all_ckpts = self.checkpoint_history["checkpoints"]
        best_paths = {cp["path"] for cp in self.checkpoint_history.get("best_checkpoints", [])}

        if len(all_ckpts) <= self.config.keep_last_n:
            return

        to_remove = all_ckpts[: -self.config.keep_last_n]
        for ckpt in to_remove:
            p = Path(ckpt["path"])
            if p.as_posix() not in best_paths and p.exists():
                p.unlink()
                self.training_logger.info(f"Removed old checkpoint: {p}")

        self.checkpoint_history["checkpoints"] = all_ckpts[-self.config.keep_last_n :]

    def load_checkpoint(
        self,
        *,
        params_template=None,
        opt_state_template=None,
    ) -> Optional[dict]:
        """Load a checkpoint based on ``config.resume_option``.

        Args:
            params_template: Unused in the pickle backend
                (kept for API parity with the Torch manager).
            opt_state_template: Unused (same reason).

        Returns:
            Dict with keys ``step``, ``epoch``, ``params``,
            ``opt_state`` (may be *None*), ``rng``, ``metrics``
            — or *None* if no checkpoint is found.
        """
        option = self.config.resume_option
        if option == "last":
            return self._load_latest()
        if option == "best":
            return self._load_best()
        if option == "path":
            if not self.config.checkpoint_path:
                raise ValueError("checkpoint_path must be set when " "resume_option='path'")
            return self._load_from_path(self.config.checkpoint_path)
        raise ValueError(f"Unknown resume_option: {option}")

    def _load_from_path(self, path: str) -> dict:
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self.training_logger.info(f"Loaded checkpoint from {path}")
        return {
            "step": data.get("step", 0),
            "epoch": data.get("epoch", 0),
            "params": data.get("params"),
            "opt_state": data.get("opt_state"),
            "rng": data.get("rng"),
            "metrics": data.get("metrics", {}),
        }

    def _load_latest(self) -> Optional[dict]:
        ckpts = self.checkpoint_history["checkpoints"]
        if not ckpts:
            return None
        return self._load_from_path(ckpts[-1]["path"])

    def _load_best(self) -> Optional[dict]:
        best = self.checkpoint_history.get("best_checkpoints", [])
        if not best:
            self.training_logger.warning("No best checkpoints found")
            return None
        return self._load_from_path(best[0]["path"])

    def get_checkpoint_info(self) -> dict:
        """Return a summary of tracked checkpoints."""
        ckpts = self.checkpoint_history["checkpoints"]
        return {
            "total_checkpoints": len(ckpts),
            "best_checkpoints": self.checkpoint_history.get("best_checkpoints", []),
            "latest_checkpoint": (ckpts[-1] if ckpts else None),
        }

    def __repr__(self) -> str:
        n = len(self.checkpoint_history["checkpoints"])
        return f"JaxCheckpointManager(" f"dir={self.checkpoint_dir!r}, " f"tracked={n})"
