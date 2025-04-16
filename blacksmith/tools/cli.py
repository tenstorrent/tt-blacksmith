# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import yaml
from pydantic import BaseModel


def generate_config(config: BaseModel, yaml_path):
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return config.model_validate(data)


def print_trainable_params(model):
    """Helper function for lora models to check number of trainable parameters."""
    total_params = sum([p.numel() for p in model.parameters()])
    trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(
        f"""
    {total_params} total params,
    {trainable_params}" trainable params,
    {(100.0 * trainable_params / total_params):.2f}% of all params are trainable.
    """
    )
