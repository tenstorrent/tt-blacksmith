# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from string import Template

PROMPT_TEMPLATE = Template(
    """
Below is an instruction from a human that describes a task. Write a response that appropriately completes the request in a gpt-like format.

### Human:
$input

### GPT:

"""
)

DATASET_PATH = "FuseAI/FuseChat-3.0-SFT-Data"
