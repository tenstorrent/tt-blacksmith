# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset

from blacksmith.datasets.fusechat_utils import DATASET_PATH, PROMPT_TEMPLATE
from blacksmith.datasets.llm_dataset import LLMDataset


class FuseChatDataset(LLMDataset):
    # FuseChat only has a train split; validation is carved out of it.
    train_val_split_ratio = 0.98

    def _load_raw_dataset(self):
        return load_dataset(DATASET_PATH, split="train")

    def _prompt_and_response(self, example):
        conversation = example["conversations"]
        return PROMPT_TEMPLATE.substitute(input=conversation[0]["value"]), conversation[1]["value"]
