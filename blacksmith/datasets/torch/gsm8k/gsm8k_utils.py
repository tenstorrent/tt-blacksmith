# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Constants, parsing helpers, and GRPO reward scoring for GSM8K.
"""
import re
from typing import List, Tuple

import torch

# Dataset source
DATASET_PATH = "openai/gsm8k"
DATASET_CONFIG = "main"

# R1-style system prompt. Gemma chat templates do not support a system role, so
# this is folded into the user turn. It instructs the model to emit its reasoning
# and final answer inside the tags scored by the format reward.
GSM8K_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process in "
    "the mind and then provides the user with the answer. The reasoning process and "
    "answer are enclosed within <reasoning> </reasoning> and <answer> </answer> tags, "
    "respectively, i.e., <reasoning> reasoning process here </reasoning> "
    "<answer> answer here </answer>. The answer must be a single integer."
)

_NUMBER_PATTERN = re.compile(r"-?\d+")

# A completion is well-formatted when a reasoning block is immediately followed
# by an answer block (matches the tags required by ``GSM8K_SYSTEM_PROMPT``).
_FORMAT_PATTERN = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.DOTALL)


def _last_number(text: str) -> str:
    """Return the last integer found in ``text`` (commas removed), or ""."""
    matches = _NUMBER_PATTERN.findall(text.replace(",", ""))
    return matches[-1] if matches else ""


def extract_gsm8k_gold(answer: str) -> str:
    """Parse the gold integer from a GSM8K ``answer`` field (``... #### 42``)."""
    if "####" in answer:
        answer = answer.split("####")[-1]
    return _last_number(answer)


def extract_xml_answer(text: str) -> str:
    """Return the text between the first ``<answer>`` and ``</answer>`` tags."""
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    return ""


def extract_predicted_answer(text: str) -> str:
    """Best-effort extraction of the model's integer answer.

    Prefers the last number inside the ``<answer>`` tags; otherwise falls back to
    the last number anywhere in the completion (so a base model can still be scored).
    """
    xml = extract_xml_answer(text)
    if xml:
        num = _last_number(xml)
        if num:
            return num
    return _last_number(text)


def format_reward(completion: str) -> float:
    """1.0 if the completion matches the reasoning/answer tag format, else 0.0."""
    return 1.0 if _FORMAT_PATTERN.search(completion) else 0.0


def correctness_reward(completion: str, gold: str) -> float:
    """1.0 if the extracted answer equals the gold answer, else 0.0."""
    pred = extract_predicted_answer(completion)
    return 1.0 if pred != "" and pred == gold else 0.0


def compute_rewards(
    completions: List[str],
    golds: List[str],
    format_weight: float = 1.0,
    correct_weight: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score a list of completions.

    Returns ``(total_rewards, format_flags, correct_flags)`` as CPU float tensors
    of shape ``(len(completions),)``. Weights follow the referenced blog (format
    up to 1.0, correctness up to 2.0), so total rewards lie in [0, 3].
    """
    format_flags = torch.tensor([format_reward(c) for c in completions], dtype=torch.float32)
    correct_flags = torch.tensor([correctness_reward(c, g) for c, g in zip(completions, golds)], dtype=torch.float32)
    total = format_weight * format_flags + correct_weight * correct_flags
    return total, format_flags, correct_flags
