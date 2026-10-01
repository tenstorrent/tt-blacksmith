# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from string import Template

from datasets import load_dataset

PROMPT_INTRO = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

PROMPT_TEMPLATE = Template(
    f"""
{PROMPT_INTRO}

### Instruction:
$instruction

### Input:
$input

### Response:
"""
)

PROMPT_TEMPLATE_NO_INPUT = Template(
    f"""
{PROMPT_INTRO}

### Instruction:
$instruction

### Response:
"""
)

DATASET_PATH = "tatsu-lab/alpaca"
from blacksmith.datasets.llm_dataset import LLMDataset


class AlpacaDataset(LLMDataset):
    # Alpaca only has a train split; validation is carved out of it.
    train_val_split_ratio = 0.98

    def _load_raw_dataset(self):
        return load_dataset(DATASET_PATH, split="train")

    def _prompt_and_response(self, example):
        instruction = example["instruction"]
        input_text = example.get("input", "")
        # Use different template based on whether there's an input field.
        if input_text.strip():
            prompt = PROMPT_TEMPLATE.substitute(instruction=instruction, input=input_text)
        else:
            prompt = PROMPT_TEMPLATE_NO_INPUT.substitute(instruction=instruction)
        return prompt, example["output"]
