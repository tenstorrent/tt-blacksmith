# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import yaml
from pydantic import BaseModel


def generate_config(config: BaseModel, yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return config.model_validate(data)
