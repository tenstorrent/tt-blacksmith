# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Dict, Optional

from blacksmith.datasets.torch.custom.templates.alpaca import build_alpaca_prompt


class AvailableTemplates(Enum):
    ALPACA = "alpaca"


TEMPLATE_KEYS = {AvailableTemplates.ALPACA.value: {"required": {"instruction", "output"}, "optional": {"input"}}}


def build_prompt(example: Dict, template: str, column_mapping: Dict) -> tuple[str, str, str]:
    template = template.strip().lower()

    if template == AvailableTemplates.ALPACA.value:
        return build_alpaca_prompt(example, column_mapping)
    else:
        available_templates = [f.value for f in AvailableTemplates]
        raise ValueError(
            f"Selected template is unsupported:  {template}. You should use one of the available templates: {available_templates}"
        )


def normalize_file_type(file_type: str) -> str:
    if file_type == "jsonl":
        file_type = "json"
    return file_type


def resolve_column_mapping(
    template_name: str, column_mapping: Optional[Dict[str, str]], dataset_columns: set[str]
) -> Dict[str, str]:
    if template_name not in TEMPLATE_KEYS:
        available_templates = [f.value for f in AvailableTemplates]
        raise ValueError(
            f"Selected template is unsupported:  {template_name}. "
            f"You should use one of the available templates: {available_templates}"
        )

    required_keys = TEMPLATE_KEYS[template_name]["required"]
    optional_keys = TEMPLATE_KEYS[template_name]["optional"]
    all_keys = required_keys.union(optional_keys)

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
        extra_keys = set(resolved.keys()) - all_keys
        if extra_keys:
            raise ValueError(
                f"Column mapping contains unsupported keys: {sorted(extra_keys)}. "
                f"Supported keys for template '{template_name}': {sorted(all_keys)}."
            )

    # Fill required keys by identity if possible
    for key in required_keys:
        if key in dataset_columns and key not in resolved:
            resolved[key] = key

    # Fill optional keys by identity if present in dataset columns
    for key in optional_keys:
        if key in dataset_columns and key not in resolved:
            resolved[key] = key

    # Final required check
    still_missing_required = required_keys - set(resolved.keys())
    if still_missing_required:
        raise ValueError(
            f"Column mapping is missing required keys: {sorted(still_missing_required)}. "
            f"Required keys for template '{template_name}': {sorted(required_keys)}."
        )

    return resolved
