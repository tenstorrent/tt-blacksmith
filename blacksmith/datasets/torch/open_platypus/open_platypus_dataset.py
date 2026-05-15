# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Open-Platypus instruction-following dataset.

Open-Platypus (`garage-bAInd/Open-Platypus`) is a 24,926-row reasoning-focused
instruction-tuning corpus distilled from 11 sources (MATH, ScienceQA, ReClor,
TheoremQA, ARB, ...). Each row exposes the Stanford-Alpaca schema:

    {
        "instruction": str,
        "input": str,          # often empty
        "output": str,
        "data_source": str,    # source-of-record tag, ignored here
    }

Because the field names are identical to Alpaca, this module reuses the same
two prompt-rendering paths controlled by ``config.prompt_format``:

    * ``"alpaca"`` — legacy ``### Instruction / ### Response`` plain-text
      template (matches the reference Platypus training code at
      https://github.com/arielnlee/Platypus).
    * ``"chat"``   — route through ``tokenizer.apply_chat_template`` so that
      an instruction-tuned checkpoint (e.g. ``google/gemma-4-E2B-it``) sees the
      exact turn-boundary tokens it was post-trained with.

Only the assistant content is supervised: the prompt half of ``labels`` is
overwritten with ``-100`` so loss is taken on the response only. The
``data_source`` column is not surfaced to the model — it is only useful for
post-hoc per-source analysis and would otherwise leak the answer's origin
into the input.
"""
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

DATASET_PATH = "garage-bAInd/Open-Platypus"

TRAIN_VAL_SPLIT_RATIO = 0.98


class OpenPlatypusDataset(BaseDataset):
    """LoRA-friendly Open-Platypus dataset.

    Open-Platypus ships only a ``train`` split, so we carve a deterministic
    98/2 train/validation slice from a single shuffled view (cached on the
    class so a paired ``train`` + ``validation`` instantiation does not
    re-tokenize the corpus).
    """

    # Open-Platypus dataset only has train split, so we create validation/test from it.
    # Cached on the class so pairing `train` + `validation` instantiations does
    # not re-tokenize the corpus (mirrors AlpacaDataset / MetaMathQADataset).
    _shared_dataset = None

    def __init__(self, config: TrainingConfig, split: str = "train", collate_fn=None):
        """
        Args:
            config: TrainingConfig (ensure ``config.dataset_id`` is set to
                ``"open_platypus"``).
            split: ``"train"`` or ``"validation"``.
            collate_fn: Optional secondary collate (applied AFTER
                ``DataCollatorForSeq2Seq``, e.g.
                ``collate_fn_for_causal_lm`` to pre-shift labels).
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
            prompt, full_text = self._render_alpaca(instruction, input_text, output)

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

    def _render_alpaca(self, instruction: str, input_text: str, output: str):
        """Legacy Stanford/Platypus plain-text template (`### Instruction / ### Response`)."""
        if input_text.strip():
            prompt = PROMPT_TEMPLATE.substitute(instruction=instruction, input=input_text)
        else:
            prompt = PROMPT_TEMPLATE_NO_INPUT.substitute(instruction=instruction)
        full_text = prompt + output
        return prompt, full_text

    def _render_chat(self, instruction: str, input_text: str, output: str):
        """Render via the tokenizer's chat template so the prompt boundary tokens
        match what the `-it` checkpoint was post-trained with.

        The Open-Platypus ``input`` field is treated as auxiliary context for
        the instruction and concatenated into the user turn (``instruction\\n\\ninput``);
        ``data_source`` is intentionally dropped — it is metadata, not part of
        the user turn. Returns ``(prompt, full_text)`` strings, where
        ``prompt`` ends right before the assistant content (so that
        ``len(tokenize(prompt))`` is the correct mask boundary) and
        ``full_text`` contains the assistant turn closed by the template's own
        end-of-turn marker — no manual EOS append (e.g. Gemma-4 closes with
        ``<end_of_turn>\\n``).
        """
        user_content = f"{instruction}\n\n{input_text}".strip() if input_text.strip() else instruction

        messages = []
        if self.config.chat_system_prompt:
            messages.append({"role": "system", "content": self.config.chat_system_prompt})
        messages.append({"role": "user", "content": user_content})

        # NOTE: ``enable_thinking=False`` is *required* for Gemma 4 -it — see
        # the matching note in ``WizardLMEvolDataset._render_chat``. Without
        # it, the template inserts a ``<think>`` opener into the prompt half
        # only, which shifts the ``labels[:prompt_len] = -100`` boundary and
        # silently masks the first supervised-output tokens.
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": output}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        return prompt, full_text

    def _prepare_dataset(self):
        if OpenPlatypusDataset._shared_dataset is None:
            raw_dataset = load_dataset(DATASET_PATH, split="train")
            tokenized_dataset = raw_dataset.map(self._tokenize_function)
            filtered_dataset = tokenized_dataset.filter(lambda x: x["len"] <= self.config.max_length)
            filtered_dataset = filtered_dataset.remove_columns(
                [col for col in filtered_dataset.column_names if col not in self.required_columns]
            )
            filtered_dataset = filtered_dataset.shuffle(seed=self.config.seed)
            OpenPlatypusDataset._shared_dataset = filtered_dataset

        full_dataset = OpenPlatypusDataset._shared_dataset
        n = len(full_dataset)
        train_end = int(TRAIN_VAL_SPLIT_RATIO * n)
        if self.split == "train":
            self.dataset = full_dataset.select(range(0, train_end))
        elif self.split == "validation":
            self.dataset = full_dataset.select(range(train_end, n))
        else:
            raise ValueError(
                f"Invalid split '{self.split}' for OpenPlatypusDataset. "
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
