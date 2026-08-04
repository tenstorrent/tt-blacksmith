# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Dict, Optional

from blacksmith.datasets.torch.custom.templates.alpaca import build_alpaca_prompt


class AvailableFormats(Enum):
    ALPACA = "alpaca"


FORMAT_KEYS = {AvailableFormats.ALPACA.value: {"required": {"instruction", "output"}, "optional": {"input"}}}


def build_prompt(example: Dict, format: str, column_mapping: Dict) -> tuple[str, str, str]:
    format = format.strip().lower()

    if format == AvailableFormats.ALPACA.value:
        return build_alpaca_prompt(example, column_mapping)
    else:
        available_formats = [f.value for f in AvailableFormats]
        raise ValueError(
            f"Selected format is unsupported: {format}. You should use one of the available formats: {available_formats}"
        )


def normalize_file_type(file_type: str) -> str:
    if file_type == "jsonl":
        file_type = "json"
    return file_type


def resolve_column_mapping(
    format_name: str, column_mapping: Optional[Dict[str, str]], dataset_columns: set[str]
) -> Dict[str, str]:
    if format_name not in FORMAT_KEYS:
        available_formats = [f.value for f in AvailableFormats]
        raise ValueError(
            f"Selected format is unsupported: {format_name}. "
            f"You should use one of the available formats: {available_formats}"
        )

    required_keys = FORMAT_KEYS[format_name]["required"]
    optional_keys = FORMAT_KEYS[format_name]["optional"]
    all_possible_keys = required_keys.union(optional_keys)

    resolved = column_mapping.copy() if column_mapping else {}

    if resolved:
        # Check provided mapping values exist in dataset columns
        missing_keys = set(resolved.values()) - dataset_columns
        if missing_keys:
            raise ValueError(
                f"Column mapping refers to non-existent dataset columns: {sorted(missing_keys)}. "
                f"Dataset columns: {sorted(dataset_columns)}."
            )

        # Check provided mapping keys are supported
        extra_keys = set(resolved.keys()) - all_possible_keys
        if extra_keys:
            raise ValueError(
                f"Column mapping contains unsupported keys: {sorted(extra_keys)}. "
                f"Supported keys for format '{format_name}': {sorted(all_possible_keys)}."
            )

    # Fill required keys by identity if possible
    for key in required_keys:
        if key in dataset_columns and key not in resolved:
            resolved[key] = key

    # Fill optional keys by identity if present
    for key in optional_keys:
        if key in dataset_columns and key not in resolved:
            resolved[key] = key

    # Final required check
    still_missing_required = required_keys - set(resolved.keys())
    if still_missing_required:
        raise ValueError(
            f"Column mapping is missing required keys: {sorted(still_missing_required)}. "
            f"Required keys for format '{format_name}': {sorted(required_keys)}."
        )

    return resolved
