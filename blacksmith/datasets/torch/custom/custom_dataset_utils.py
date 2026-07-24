# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from string import Template

ALPACA_PROMPT_INTRO = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)
ALPACA_PROMPT_TEMPLATE = Template(
    f"""
{ALPACA_PROMPT_INTRO}

### Instruction:
$instruction

$input_section

### Response:
"""
)
ALPACA_INPUT_BLOCK_TEMPLATE = Template(
    """### Input:
$input
"""
)


def build_alpaca_prompt(instruction: str, input_text: str = "") -> str:
    input_section = ALPACA_INPUT_BLOCK_TEMPLATE.substitute(input=input_text) if input_text.strip() else ""
    return ALPACA_PROMPT_TEMPLATE.substitute(
        instruction=instruction,
        input_section=input_section,
    )
