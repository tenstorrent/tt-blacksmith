# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from datasets import load_dataset

from blacksmith.configs.trainer import TrainerConfig
from blacksmith.datasets.custom_utils import (
    build_prompt,
    normalize_file_type,
    resolve_column_mapping,
)
from blacksmith.datasets.llm_dataset import LLMDataset


class CustomLLMDataset(LLMDataset):
    def __init__(self, config: TrainerConfig, split: str = "train", collate_fn=None):
        self.file_type = normalize_file_type(config.custom_dataset.file_type)
        self.data_path = (
            config.custom_dataset.train_dataset_path if split == "train" else config.custom_dataset.val_dataset_path
        )
        self.template = config.custom_dataset.template
        self.column_mapping = config.custom_dataset.column_mapping
        super().__init__(config, split, collate_fn)

    def _load_raw_dataset(self):
        if not self.data_path:
            if self.split == "train":
                raise ValueError("train_dataset_path is required and was not provided.")
            return None
        raw_dataset = load_dataset(self.file_type, data_files={self.split: self.data_path}, split=self.split)
        self.column_mapping = resolve_column_mapping(self.template, self.column_mapping, set(raw_dataset[0].keys()))
        return raw_dataset

    def _prompt_and_response(self, example):
        prompt, output, _ = build_prompt(example, template=self.template, column_mapping=self.column_mapping)
        return prompt, output

    def _prepare_dataset(self):
        super()._prepare_dataset()
        if self.split == "train" and self.dataset is not None:
            self.dataset = self.dataset.shuffle(seed=self.config.seed)
