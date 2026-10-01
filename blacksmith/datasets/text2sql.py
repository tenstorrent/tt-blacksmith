# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import logging
from string import Template

from datasets import load_dataset

from blacksmith.configs import TrainerConfig

PROMPT_TEMPLATE = Template(
    """### Instruction:\n
Generate an SQL query for the given question and database schema.\n\n
### Input:\n
Question: $prompt\n
Schema: $context\n\n
### Output:\n"""
)
DATASET_PATH = "gretelai/synthetic_text_to_sql"

logger = logging.getLogger(__name__)
from blacksmith.datasets.llm_dataset import LLMDataset


class TextToSQLDataset(LLMDataset):
    def __init__(self, config: TrainerConfig, split: str = "train", collate_fn=None):
        # This dataset only has "train"/"test" splits.
        if split == "validation":
            logger.warning("Validation split does not exist for TextToSQLDataset, defaulting to test split.")
            split = "test"
        super().__init__(config, split, collate_fn)

    def _load_raw_dataset(self):
        raw_dataset = load_dataset(DATASET_PATH, split=self.split)
        return raw_dataset.filter(lambda example: example["sql_complexity"] == "basic SQL")

    def _prompt_and_response(self, example):
        prompt = PROMPT_TEMPLATE.substitute(prompt=example["sql_prompt"], context=example.get("sql_context", ""))
        return prompt, example["sql"].strip()
