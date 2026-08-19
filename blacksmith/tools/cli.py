# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

_TEST_MODE_DEFAULTS = {
    "test_config": {"max_steps_per_epoch": 15},
    "steps_freq": 5,
    "val_steps_freq": 5,
    "save_strategy": "none",
    "use_tt": True,
    "log_on_wandb": False,
}


def _apply_overrides(config_data: dict, overrides: list[str]) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw_value = item.partition("=")
        key = key.strip()
        try:
            value = yaml.safe_load(raw_value.strip())
        except yaml.YAMLError as e:
            raise ValueError(f"--set {key}: could not parse {raw_value.strip()!r}") from e

        target = config_data
        *parents, leaf = key.split(".")
        for part in parents:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"--set {key}: {part!r} is not a nested section")
        target[leaf] = value
    return config_data


def generate_config(
    config: BaseModel,
    yaml_path: Path,
    test_yaml_path: Optional[Path] = None,
    test_checkpoint_path: Optional[str] = None,
    reference_model_checkpoint_path: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> BaseModel:
    assert yaml_path.exists(), f"Config file {yaml_path} does not exist"
    with yaml_path.open() as file:
        config_data = yaml.safe_load(file)

    # When running under pytest, apply defaults to limit training duration and
    # logging frequency. An explicit test config (below) can still override.
    if "PYTEST_CURRENT_TEST" in os.environ:
        config_data |= _TEST_MODE_DEFAULTS

    if test_yaml_path is not None:
        # This enables test config to overwrite some fields in original config or add new ones for example `test_config`.
        assert test_yaml_path.exists(), f"Test config file {yaml_path} does not exist"
        with test_yaml_path.open() as file:
            config_data |= yaml.safe_load(file)

    if test_checkpoint_path:
        config_data["resume_from_checkpoint"] = True
        config_data["resume_option"] = "path"
        config_data["checkpoint_path"] = test_checkpoint_path

    if reference_model_checkpoint_path:
        config_data["sft_checkpoint_path"] = reference_model_checkpoint_path

    if overrides:
        config_data = _apply_overrides(config_data, overrides)

    return config.model_validate(config_data)


def parse_cli_options(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Experiment CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    if default_config.is_relative_to(Path.cwd()):
        default_config = default_config.relative_to(Path.cwd())

    parser.add_argument("--config", type=Path, default=default_config, help="Path to YAML config file")

    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="Override one config value, e.g. --set max_steps=100 or --set inference.infer_steps=10 (repeatable)",
    )

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
