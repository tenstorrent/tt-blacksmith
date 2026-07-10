# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from string import Template

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from blacksmith.datasets.torch.torch_dataset import BaseDataset
from blacksmith.tools.templates.configs import TrainingConfig
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


class WizardLMEvolDataset(BaseDataset):
    # WizardLM-Evol-Instruct only has train split, so we create validation from it.
    _shared_dataset = None

    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        """
        Args:
            config: TrainingConfig (ensure config.dataset_id is set to "wizardlm_evol")
            split: Dataset split to use ("train" or "validation")
            collate_fn: Collate function to use for the dataset
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side="right", use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.required_columns = ["input_ids", "attention_mask", "labels"]

        super().__init__(config, split, collate_fn)

    def _tokenize_function(self, example):
        instruction = example["instruction"]
        input_text = example.get("input", "") or ""
        output = example["output"]

        if self.config.prompt_format == "chat":
            prompt, full_text = self._render_chat(instruction, input_text, output)
        else:
            prompt, full_text = self._render_default(instruction, input_text, output)

        encoding = self.tokenizer(full_text, truncation=False, padding=False, return_tensors="pt")

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        prompt_encoding = self.tokenizer(prompt, truncation=False, padding=False, return_tensors="pt")
        prompt_input_ids = prompt_encoding["input_ids"].squeeze(0)
        prompt_len = prompt_input_ids.size(0)
        labels[:prompt_len] = -100

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["labels"] = labels
        example["full_text"] = full_text
        example["len"] = input_ids.size(0)

        return example

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

    def _prepare_dataset(self):
        if WizardLMEvolDataset._shared_dataset is None:
            raw_dataset = load_dataset(DATASET_PATH, split="train")
            tokenized_dataset = raw_dataset.map(self._tokenize_function)
            filtered_dataset = tokenized_dataset.filter(lambda x: x["len"] <= self.config.max_length)
            filtered_dataset = filtered_dataset.remove_columns(
                [col for col in filtered_dataset.column_names if col not in self.required_columns]
            )
            filtered_dataset = filtered_dataset.shuffle(seed=self.config.seed)
            WizardLMEvolDataset._shared_dataset = filtered_dataset

        full_dataset = WizardLMEvolDataset._shared_dataset
        n = len(full_dataset)
        train_end = int(TRAIN_VAL_SPLIT_RATIO * n)
        if self.split == "train":
            self.dataset = full_dataset.select(range(0, train_end))
        elif self.split == "validation":
            self.dataset = full_dataset.select(range(train_end, n))
        else:
            raise ValueError(
                f"Invalid split '{self.split}' for WizardLMEvolDataset. "
                "Only 'train' and 'validation' are supported."
            )

    def __len__(self):
        return len(self.dataset)

    def _get_dataloader(self) -> DataLoader:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, padding="max_length", max_length=self.config.max_length
        )

        if self.collate_fn is not None:
            total_collate_fn = lambda batch: self.collate_fn(data_collator(batch))
        else:
            total_collate_fn = data_collator

        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=total_collate_fn,
            shuffle=self.split == "train",
            drop_last=True,
        )
