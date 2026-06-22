# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Constants and utilities for DPO (Direct Preference Optimization) dataset.

Based on the paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
https://arxiv.org/pdf/2305.18290
"""
from enum import Enum
from string import Template


class DatasetMode(Enum):
    """Mode for the math preference dataset."""

    DPO = "dpo"  # Returns chosen + rejected for DPO training.
    SFT = "sft"  # Returns only chosen for SFT training.


# Fraction of each mode-specific pool used for the train split (rest is validation).
TRAIN_VAL_SPLIT_RATIO = 0.9
# Default fraction in [0, 1] used to split filtered train data into SFT and DPO pools.
DEFAULT_SFT_RATIO = 0.33

# Dataset source.
DATASET_PATH = "argilla/distilabel-math-preference-dpo"
DATASET_NAME = "distilabel-math-preference-dpo"

# Prompt template for math instructions.
PROMPT_TEMPLATE = Template("### Instruction:\n$instruction\n\n### Response:\n")

# Label for masking prompt tokens in loss computation.
IGNORED_LABEL_ID = -100

# Required columns for DPO training.
DPO_REQUIRED_COLUMNS = [
    "chosen_input_ids",
    "chosen_attention_mask",
    "chosen_labels",
    "rejected_input_ids",
    "rejected_attention_mask",
    "rejected_labels",
]

# Required columns for SFT training.
SFT_REQUIRED_COLUMNS = [
    "input_ids",
    "attention_mask",
    "labels",
]

# Original dataset columns.
SOURCE_COLUMNS = [
    "instruction",
    "chosen_response",
    "rejected_response",
    "chosen_rating",
    "rejected_rating",
]
