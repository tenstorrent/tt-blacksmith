# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from inspect import cleandoc
from string import Template

from datasets import load_dataset

PROMPT_TEMPLATE = Template(
    cleandoc(
        """
        Below is an instruction that describes a task.
        Write a response that appropriately completes the request.

        ### Instruction:
        $instruction

        ### Response:
        $response
        """
    )
)

DATASET_PATH = "meta-math/MetaMathQA"

TRAIN_VAL_SPLIT_RATIO = 0.98
from blacksmith.datasets.llm_dataset import LLMDataset


class MetaMathQADataset(LLMDataset):
    # MetaMathQA only has a train split; validation is carved out of it.
    train_val_split_ratio = TRAIN_VAL_SPLIT_RATIO

    def _load_raw_dataset(self):
        return load_dataset(DATASET_PATH, split="train")

    def _prompt_and_response(self, example):
        prompt = PROMPT_TEMPLATE.substitute(instruction=example["query"], response="")
        return prompt, example["response"]
