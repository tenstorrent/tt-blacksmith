# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from string import Template

from datasets import load_dataset

PROMPT_INTRO = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
)

PROMPT_TEMPLATE = Template(
    f"""
{PROMPT_INTRO}

### Instruction:
$instruction

### Input:
$input

### Response:
"""
)

PROMPT_TEMPLATE_NO_INPUT = Template(
    f"""
{PROMPT_INTRO}

### Instruction:
$instruction

### Response:
"""
)

DATASET_PATH = "WizardLMTeam/WizardLM_evol_instruct_70k"

TRAIN_VAL_SPLIT_RATIO = 0.98
from blacksmith.datasets.llm_dataset import LLMDataset


class WizardLMEvolDataset(LLMDataset):
    # WizardLM-Evol-Instruct only has a train split; validation is carved out of it.
    train_val_split_ratio = TRAIN_VAL_SPLIT_RATIO

    def _load_raw_dataset(self):
        return load_dataset(DATASET_PATH, split="train")

    def _render(self, example):
        instruction = example["instruction"]
        input_text = example.get("input", "") or ""
        output = example["output"]
        if self.config.prompt_format == "chat":
            return self._render_chat(instruction, input_text, output)
        return self._render_default(instruction, input_text, output)

    # Plain-text instruction template (`### Instruction / ### Input / ### Response`).
    # Use for base (non `-it`) checkpoints.
    def _render_default(self, instruction: str, input_text: str, output: str):
        if input_text.strip():
            prompt = PROMPT_TEMPLATE.substitute(instruction=instruction, input=input_text)
        else:
            prompt = PROMPT_TEMPLATE_NO_INPUT.substitute(instruction=instruction)
        full_text = prompt + output
        return prompt, full_text

    # Route through the tokenizer's chat template (model-specific turn-boundary
    # tokens, e.g. Gemma-4 `<start_of_turn>...<end_of_turn>`). Use for `-it`
    # checkpoints so training matches the post-training token sequence.
    def _render_chat(self, instruction: str, input_text: str, output: str):
        user_content = f"{instruction}\n\n{input_text}".strip() if input_text.strip() else instruction
        messages = []
        if self.config.chat_system_prompt:
            messages.append({"role": "system", "content": self.config.chat_system_prompt})
        messages.append({"role": "user", "content": user_content})
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full_text = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": output}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return prompt, full_text
