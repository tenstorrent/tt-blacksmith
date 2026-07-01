# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GSM8K dataset for GRPO (Group Relative Policy Optimization) training.

Unlike the supervised datasets in this package (which return tokenized
``input_ids``/``labels`` tensors), GRPO trains directly on prompts: the policy
model samples several completions per prompt and is rewarded with user-defined
reward functions. This loader therefore exposes the raw Hugging Face
``datasets.Dataset`` (via ``self.dataset``) in the prompt-completion format
expected by TRL's ``GRPOTrainer``:

    {
        "prompt": [{"role": "user", "content": <R1-style instructions + question>}],
        "answer": "<ground-truth integer as a string>",
    }

The ``answer`` column is not consumed by the trainer directly; TRL forwards every
extra dataset column to the reward functions as a keyword argument, so the
correctness reward can compare a completion against the ground truth.

Source: https://huggingface.co/datasets/openai/gsm8k
"""
from inspect import cleandoc
from typing import Dict, Optional

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig
from datasets import load_dataset

DATASET_PATH = "openai/gsm8k"
DATASET_CONFIG = "main"

# GSM8K answers store the final value after a "####" delimiter.
ANSWER_DELIMITER = "####"

# R1-style prompt used in the GRPO blog. Gemma 1.1 has no dedicated system role,
# so these instructions are prepended to the user turn instead.
SYSTEM_PROMPT = cleandoc(
    """
    A conversation between User and Assistant. The user asks a question, and the
    Assistant solves it. The assistant first thinks about the reasoning process
    in the mind and then provides the user with the answer.

    The reasoning process and answer are enclosed within <reasoning></reasoning>
    and <answer></answer> tags. The answer must be a single integer, for example:
    <reasoning>step-by-step reasoning here</reasoning>
    <answer>42</answer>
    """
)


def extract_hash_answer(answer: str) -> Optional[str]:
    """Return the ground-truth integer string after the GSM8K '####' delimiter."""
    if ANSWER_DELIMITER not in answer:
        return None
    return answer.split(ANSWER_DELIMITER)[-1].strip().replace(",", "").replace("$", "")


class GSM8KDataset(BaseDataset):
    """GSM8K grade-school math dataset in TRL prompt-completion format."""

    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        """
        Args:
            config: TrainingConfig (ensure ``config.dataset_id`` is set to "gsm8k").
            split: Dataset split to use ("train" or "test").
            collate_fn: Unused; kept for API compatibility with the dataset factory.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        super().__init__(config, split, collate_fn)

    def _build_prompt(self, question: str) -> list[Dict[str, str]]:
        """Wrap a question into the conversational prompt TRL expects."""
        return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{question}"}]

    def _format_example(self, example: Dict) -> Dict:
        return {
            "prompt": self._build_prompt(example["question"]),
            "answer": extract_hash_answer(example["answer"]),
        }

    def _prepare_dataset(self):
        if self.split not in ("train", "test"):
            raise ValueError(f"Invalid split '{self.split}' for GSM8KDataset. Only 'train' and 'test' are supported.")

        raw_dataset = load_dataset(DATASET_PATH, DATASET_CONFIG, split=self.split)
        formatted = raw_dataset.map(self._format_example, remove_columns=raw_dataset.column_names)
        formatted = formatted.filter(lambda example: example["answer"] is not None)

        if self.split == "train":
            formatted = formatted.shuffle(seed=self.config.seed)

        self.dataset = formatted

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx]

    def _get_dataloader(self) -> DataLoader:
        """Return a plain DataLoader over prompts.

        GRPO training is driven by TRL's ``GRPOTrainer``, which consumes the raw
        ``datasets.Dataset`` exposed as ``self.dataset`` and handles its own
        batching/generation. This method exists to satisfy ``BaseDataset`` and is
        not used on the GRPO training path.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.split == "train",
        )
