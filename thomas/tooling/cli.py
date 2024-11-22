# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, fields, is_dataclass
import yaml


def from_dict(cls, data):
    if not is_dataclass(cls):
        return data  # Not a dataclass, return as-is

    field_types = {f.name: f.type for f in fields(cls)}
    return cls(**{key: from_dict(field_types[key], value) for key, value in data.items()})


def generate_config(config, yaml_path):
    assert is_dataclass(config), "config must be a dataclass"
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return from_dict(config, data)
