# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from blacksmith.datasets.torch.custom.templates.alpaca import (
    ALPACA_INPUT_BLOCK_TEMPLATE,
    ALPACA_PROMPT_TEMPLATE,
)


def build_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    input_section = ALPACA_INPUT_BLOCK_TEMPLATE.substitute(input=input_text) if input_text.strip() else ""
    return ALPACA_PROMPT_TEMPLATE.substitute(
        instruction=instruction,
        input_section=input_section,
    )
