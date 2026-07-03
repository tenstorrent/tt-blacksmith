# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Constants and parsing helpers for the GSM8K dataset used in GRPO training.

These are kept separate from the dataset class so the GRPO reward functions
(``blacksmith/tools/grpo_utils.py``) can reuse the same answer-extraction logic
without duplicating regexes or the prompt.
"""
import re

# Dataset source (config "main" ships train/test splits).
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

# First / last signed integer (commas stripped before matching).
_NUMBER_PATTERN = re.compile(r"-?\d+")


def _normalize_number(text: str) -> str:
    """Return the first integer found in ``text`` (commas removed), or ""."""
    match = _NUMBER_PATTERN.search(text.replace(",", ""))
    return match.group(0) if match else ""


def _last_number(text: str) -> str:
    """Return the last integer found in ``text`` (commas removed), or ""."""
    matches = _NUMBER_PATTERN.findall(text.replace(",", ""))
    return matches[-1] if matches else ""


def extract_gsm8k_gold(answer: str) -> str:
    """Parse the gold integer from a GSM8K ``answer`` field (``... #### 42``)."""
    if "####" in answer:
        answer = answer.split("####")[-1]
    return _normalize_number(answer)


def extract_xml_answer(text: str) -> str:
    """Return the text between the first ``<answer>`` and ``</answer>`` tags."""
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    return ""


def extract_predicted_answer(text: str) -> str:
    """Best-effort extraction of the model's integer answer.

    Prefers the number inside the ``<answer>`` tags; otherwise falls back to the
    last number anywhere in the completion (so a base model can still be scored).
    """
    xml = extract_xml_answer(text)
    if xml:
        num = _normalize_number(xml)
        if num:
            return num
    return _last_number(text)
