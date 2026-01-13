# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path

import yaml
from pydantic import BaseModel


def generate_config(config: BaseModel, yaml_path: Path) -> BaseModel:
    assert yaml_path.exists(), f"Config file {yaml_path} does not exist"
    with yaml_path.open() as file:
        data = yaml.safe_load(file)
    return config.model_validate(data)


def parse_cli_options(default_config: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Experiment CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--config", type=Path, default=default_config.relative_to(Path.cwd()), help="Path to YAML config file"
    )

    args = parser.parse_args()
    return args
