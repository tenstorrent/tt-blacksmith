# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Dict

from blacksmith.datasets.torch.custom.templates.alpaca import build_alpaca_prompt


class AvailableFormats(Enum):
    ALPACA = "alpaca"


FORMAT_REQUIRED_KEYS = {
    AvailableFormats.ALPACA.value: {"instruction", "output"},
}


def build_prompt(format: str, column_mapping: Dict, example: Dict):
    format = format.strip().lower()

    if format == AvailableFormats.ALPACA.value:
        return build_alpaca_prompt(column_mapping, example)
    else:
        available_formats = [f.value for f in AvailableFormats]
        raise ValueError(
            f"Selected format is unsupported: {format}. You should use one of the available formats: {available_formats}"
        )


def normalize_file_type(file_type: str) -> str:
    file_type = file_type.strip().lower()
    mapping = {
        "json": "json",
        "jsonl": "json",
    }
    if file_type not in mapping:
        raise ValueError(
            f"Selected file type is unsupported: {file_type}. "
            f"Please select one of the supported types: {list(mapping)}"
        )
    return mapping[file_type]


def validate_column_mapping(format: str, column_mapping: Dict, dataset_columns: set):
    # 1. Required keys must be present and non-empty
    required_keys = FORMAT_REQUIRED_KEYS[format]
    missing_keys = required_keys - column_mapping.keys()
    if missing_keys:
        raise ValueError(
            f"column_mapping is missing required key(s): {sorted(missing_keys)}. " f"Required: {sorted(required_keys)}."
        )
    empty_required = [k for k in required_keys if not column_mapping.get(k)]
    if empty_required:
        raise ValueError(f"Required key(s) have empty mapping: {sorted(empty_required)}.")

    # 2. Every non-empty mapped value must exist as an actual dataset column
    for col in column_mapping.values():
        if col and col not in dataset_columns:
            raise ValueError(f"Column '{col}' not found in dataset columns: {sorted(dataset_columns)}.")
