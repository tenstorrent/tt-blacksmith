# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

# Applied when an experiment runs under pytest: short runs, no W&B.
_TEST_MODE_DEFAULTS = {
    "test_config": {"max_steps_per_epoch": 15},
    "steps_freq": 5,
    "val_steps_freq": 5,
    "save_strategy": "none",
    "use_wandb": False,
}


def _deep_update(base: dict, overlay: dict) -> dict:
    """Recursively merge ``overlay`` into ``base`` so a test YAML can override
    a single nested key (e.g. ``mesh.shape``) without replacing its siblings."""
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def generate_config(
    config: type[BaseModel],
    yaml_path: Path,
    test_yaml_path: Optional[Path] = None,
) -> BaseModel:
    assert yaml_path.exists(), f"Config file {yaml_path} does not exist"
    with yaml_path.open() as file:
        config_data = yaml.safe_load(file)

    if "PYTEST_CURRENT_TEST" in os.environ:
        _deep_update(config_data, _TEST_MODE_DEFAULTS)

    if test_yaml_path is not None:
        assert test_yaml_path.exists(), f"Test config file {test_yaml_path} does not exist"
        with test_yaml_path.open() as file:
            _deep_update(config_data, yaml.safe_load(file) or {})

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

    return parser.parse_args()
