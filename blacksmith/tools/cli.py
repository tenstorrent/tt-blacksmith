# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from blacksmith.tools.trainer.configs.base import TrainerConfig

_TEST_MODE_DEFAULTS = {
    "test_config": {"max_steps_per_epoch": 15},
    "steps_freq": 5,
    "val_steps_freq": 5,
    "save_strategy": "none",
    "use_tt": True,
    "log_on_wandb": False,
}

# Nested TrainerConfig equivalents of the flat keys above. Applied under
# pytest when ``config`` is a TrainerConfig subclass; deep-merged so the
# rest of logging / metrics / checkpoint is kept.
_TRAINER_TEST_MODE_DEFAULTS = {
    "logging": {"use_wandb": False},
    "metrics": {"steps_freq": 5},
    "checkpoint": {"save_strategy": "none"},
}


def _deep_update(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into ``base``.

    Nested dicts are updated in place so a test YAML can override
    ``metrics.steps_freq`` without replacing the rest of ``metrics``.
    Non-dict values (including lists) replace.
    """
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def generate_config(
    config: BaseModel,
    yaml_path: Path,
    test_yaml_path: Optional[Path] = None,
    test_checkpoint_path: Optional[str] = None,
    reference_model_checkpoint_path: Optional[str] = None,
) -> BaseModel:
    assert yaml_path.exists(), f"Config file {yaml_path} does not exist"
    with yaml_path.open() as file:
        config_data = yaml.safe_load(file)

    # When running under pytest, apply defaults to limit training duration and
    # logging frequency. An explicit test config (below) can still override.
    if "PYTEST_CURRENT_TEST" in os.environ:
        config_data |= _TEST_MODE_DEFAULTS
        if issubclass(config, TrainerConfig):
            _deep_update(config_data, _TRAINER_TEST_MODE_DEFAULTS)

    if test_yaml_path is not None:
        # Overlay on top of the experiment YAML (and test-mode defaults).
        # Nested dicts are merged so TrainerConfig sub-configs
        # (logging / metrics / checkpoint) can be overridden field-wise.
        assert test_yaml_path.exists(), f"Test config file {test_yaml_path} does not exist"
        with test_yaml_path.open() as file:
            _deep_update(config_data, yaml.safe_load(file) or {})

    if test_checkpoint_path:
        config_data["resume_from_checkpoint"] = True
        config_data["resume_option"] = "path"
        config_data["checkpoint_path"] = test_checkpoint_path
        checkpoint = config_data.get("checkpoint")
        if isinstance(checkpoint, dict):
            checkpoint["resume_from_checkpoint"] = True
            checkpoint["resume_option"] = "path"
            checkpoint["checkpoint_path"] = test_checkpoint_path

    if reference_model_checkpoint_path:
        config_data["sft_checkpoint_path"] = reference_model_checkpoint_path

    return config.model_validate(config_data)


def parse_cli_options(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Experiment CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    if default_config.is_relative_to(Path.cwd()):
        default_config = default_config.relative_to(Path.cwd())

    parser.add_argument("--config", type=Path, default=default_config, help="Path to YAML config file")

    parser.add_argument(
        "--test-config", type=Path, required=False, help="[Testing utils] Configuration that is used for CI testing"
    )

    parser.add_argument(
        "--test-log-filename-prefix", type=str, required=False, help="[Testing utils] Prefix for the test log filename"
    )

    parser.add_argument(
        "--test-checkpoint-path", type=str, required=False, help="[Testing utils] Path to the checkpoint to resume from"
    )

    parser.add_argument(
        "--reference-model-checkpoint-path",
        type=str,
        required=False,
        help="[Testing utils] Path to the checkpoint used to initialize the DPO reference model (sft_checkpoint_path)",
    )

    args = parser.parse_args()
    return args
