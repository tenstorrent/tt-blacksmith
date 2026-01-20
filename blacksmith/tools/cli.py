# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


def generate_config(config: BaseModel, yaml_path: Path, test_yaml_path: Optional[Path] = None) -> BaseModel:
    assert yaml_path.exists(), f"Config file {yaml_path} does not exist"
    with yaml_path.open() as file:
        config_data = yaml.safe_load(file)

    if test_yaml_path is not None:
        # This enables test config to overwrite some fields in original config or add new ones for example `test_config`.
        assert test_yaml_path.exists(), f"Test config file {yaml_path} does not exist"
        with test_yaml_path.open() as file:
            config_data |= yaml.safe_load(file)

    return config.model_validate(config_data)


def parse_cli_options(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Experiment CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--config", type=Path, default=default_config.relative_to(Path.cwd()), help="Path to YAML config file"
    )

    parser.add_argument(
        "--test-config", type=Path, required=False, help="[Testing utils] Configuration that is used for CI testing"
    )

    args = parser.parse_args()
    return args
