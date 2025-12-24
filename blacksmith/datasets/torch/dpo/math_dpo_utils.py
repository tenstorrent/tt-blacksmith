# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Constants and utilities for DPO (Direct Preference Optimization) dataset.

Based on the paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/pdf/2305.18290
"""
from string import Template


# Dataset source
DATASET_PATH = "argilla/distilabel-math-preference-dpo"
DATASET_NAME = "distilabel-math-preference-dpo"

# Prompt template for math instructions
PROMPT_TEMPLATE = Template("### Instruction:\n$instruction\n\n### Response:\n")

# Label for masking prompt tokens in loss computation
IGNORED_LABEL_ID = -100

# Required columns for DPO training
REQUIRED_COLUMNS = [
    "chosen_input_ids",
    "chosen_attention_mask",
    "chosen_labels",
    "rejected_input_ids",
    "rejected_attention_mask",
    "rejected_labels",
]

# Original dataset columns
SOURCE_COLUMNS = [
    "instruction",
    "chosen_response",
    "rejected_response",
    "chosen_rating",
    "rejected_rating",
]
