# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, fields, is_dataclass
import yaml

from pydantic import BaseModel


def generate_config(config: BaseModel, yaml_path):
    assert is_dataclass(config), "config must be a dataclass"
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return config.model_validate(data)
