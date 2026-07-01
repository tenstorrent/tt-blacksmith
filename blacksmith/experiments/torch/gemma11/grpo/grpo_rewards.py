# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Reward functions for GRPO (Group Relative Policy Optimization) training on GSM8K.

Based on the GRPO method from "DeepSeekMath: Pushing the Limits of Mathematical
Reasoning in Open Language Models" (https://arxiv.org/pdf/2402.03300) and the toy
example by Luca Massaron:
- https://medium.com/@lucamassaron/training-for-reasoning-with-grpo-881e1819f2df
- https://medium.com/@lucamassaron/training-for-reasoning-with-grpo-part-ii-a-step-by-step-explanation-f80c219e2059

GRPO scores each sampled completion with one or more reward functions (no learned
reward model). Here we use two complementary, rule-based rewards:

- ``format_reward_func``: 0.0 or 1.0 depending on whether the completion follows
  the required ``<reasoning>...</reasoning><answer>...</answer>`` structure.
- ``correctness_reward_func``: 0.0 or 2.0 depending on whether the extracted
  answer matches the GSM8K ground truth.

The two rewards are summed by the trainer (0.0 .. 3.0). TRL passes each completion
in conversational form, i.e. ``completion[0]["content"]`` holds the generated text,
and forwards extra dataset columns (such as ``answer``) as keyword arguments.
"""
import re
from typing import List, Optional

FORMAT_REWARD = 1.0
CORRECTNESS_REWARD = 2.0

# Full-string match: <reasoning> ... </reasoning> <answer> ... </answer>
FORMAT_PATTERN = re.compile(r"^<reasoning>[\s\S]*?</reasoning>\s*<answer>[\s\S]*?</answer>\s*$")

# Captures the content of the (last) <answer> ... </answer> block.
ANSWER_PATTERN = re.compile(r"<answer>([\s\S]*?)</answer>")

# Any integer (optionally signed), used as a fallback when no answer tag is found.
NUMBER_PATTERN = re.compile(r"-?\d+")


def _completion_text(completion) -> str:
    """Extract the generated text from a TRL completion (conversational or plain)."""
    if isinstance(completion, str):
        return completion
    # Conversational format: a list of {"role", "content"} messages.
    return completion[0]["content"]


def extract_xml_answer(text: str) -> Optional[str]:
    """Return the normalized integer inside the last <answer> tag, if present."""
    matches = ANSWER_PATTERN.findall(text)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def extract_last_number(text: str) -> Optional[str]:
    """Return the last integer that appears anywhere in the text, if any."""
    matches = NUMBER_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1]


def _normalize_number(value: str) -> Optional[str]:
    """Strip formatting and parse a single integer from a free-form string."""
    cleaned = value.strip().replace(",", "").replace("$", "")
    match = NUMBER_PATTERN.search(cleaned)
    return match.group(0) if match else None


def format_reward_func(completions: List, **kwargs) -> List[float]:
    """Reward completions that follow the required reasoning/answer XML structure."""
    responses = [_completion_text(completion) for completion in completions]
    return [FORMAT_REWARD if FORMAT_PATTERN.match(response.strip()) else 0.0 for response in responses]


def correctness_reward_func(completions: List, answer: List[str], **kwargs) -> List[float]:
    """Reward completions whose extracted answer matches the GSM8K ground truth.

    The answer is taken from the ``<answer>`` tag when present, otherwise the last
    integer in the text is used (so even unformatted-but-correct answers count).
    """
    responses = [_completion_text(completion) for completion in completions]
    rewards = []
    for response, ground_truth in zip(responses, answer):
        extracted = extract_xml_answer(response)
        if extracted is None:
            extracted = extract_last_number(response)
        correct = _normalize_number(ground_truth) if ground_truth is not None else None
        rewards.append(CORRECTNESS_REWARD if extracted is not None and extracted == correct else 0.0)
    return rewards
