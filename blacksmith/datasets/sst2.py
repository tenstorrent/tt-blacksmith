# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset

from blacksmith.datasets.llm_dataset import LLMDataset
from blacksmith.datasets.sst2_utils import (
    DATASET_BENCHMARK,
    DATASET_NAME,
    LBL2VALUE,
    PROMPT_TEMPLATE,
    RESPONSE_TEMPLATE,
)


class SSTDataset(LLMDataset):
    def _load_raw_dataset(self):
        return load_dataset(DATASET_BENCHMARK, DATASET_NAME, split=self.split)

    def _prompt_and_response(self, example):
        prompt = PROMPT_TEMPLATE.substitute(input=example["sentence"])
        response = RESPONSE_TEMPLATE.substitute(label=LBL2VALUE[example["label"]])
        return prompt, response
