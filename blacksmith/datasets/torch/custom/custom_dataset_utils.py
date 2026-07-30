# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum

from blacksmith.datasets.torch.custom.templates.alpaca import build_alpaca_prompt


class AvailableFormats(Enum):
    ALPACA = "alpaca"


def build_prompt(format: str, instruction: str, input_text: str = ""):
    format = format.strip().lower()
    if format == AvailableFormats.ALPACA.value:
        return build_alpaca_prompt(instruction, input_text)
    else:
        available_formats = [f.value for f in AvailableFormats]
        raise ValueError(f"Unsupported format: {format}. Available formats: {available_formats}")
