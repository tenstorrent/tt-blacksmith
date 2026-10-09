# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from string import Template

from datasets import load_dataset

from blacksmith.datasets.llm_dataset import LLMDataset

PROMPT_TEMPLATE = Template(
    """
Context: $context\n
Question: $question\n
Answer:
"""
)
DATASET_PATH = "rajpurkar/squad_v2"


class SquadV2Dataset(LLMDataset):
    def _load_raw_dataset(self):
        raw_dataset = load_dataset(DATASET_PATH, split=self.split)
        # reduce the size of the validation dataset
        if self.split == "validation":
            raw_dataset = raw_dataset.train_test_split(test_size=0.02, seed=self.config.seed)["test"]
        return raw_dataset

    def _prompt_and_response(self, example):
        prompt = PROMPT_TEMPLATE.substitute(context=example["context"], question=example["question"])
        # SQuAD v2.0 has unanswerable questions, indicated by an empty 'text' list.
        response = example["answers"]["text"][0] if example["answers"]["text"] else "unanswerable"
        return prompt, response
