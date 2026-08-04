# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from string import Template
from typing import Dict

ALPACA_PROMPT_INTRO = (
    "Below is an instruction that describes a task, paired with an optional input that provides further context. "
    "Write a response that appropriately completes the request."
)

ALPACA_INPUT_BLOCK_TEMPLATE = Template(
    """### Input:
$input
"""
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


def build_alpaca_prompt(example: Dict, column_mapping: Dict) -> str:
    instruction = example[column_mapping["instruction"]]
    input_col = column_mapping.get("input", "")
    input_text = example[input_col] if input_col else ""
    output = example[column_mapping["output"]]

    input_section = ALPACA_INPUT_BLOCK_TEMPLATE.substitute(input=input_text) if input_text.strip() else ""
    prompt = ALPACA_PROMPT_TEMPLATE.substitute(instruction=instruction, input_section=input_section)

    full_text = prompt + output

    return prompt, output, full_text
