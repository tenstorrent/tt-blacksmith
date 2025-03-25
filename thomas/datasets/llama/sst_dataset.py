# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple, Dict, Any
from datasets import load_dataset
from transformers import AutoTokenizer

from thomas.datasets.llama.sst_utils import TRAIN_PROMPT_TEMPLATE, TEST_PROMPT_TEMPLATE, LBL2VALUE

from dataclasses import dataclass, asdict
import json
import os
from typing import Optional


@dataclass
class TrainingConfig:
    # Dataset settings
    dataset_id: str = "stanfordnlp/sst2"

    # Model settings
    model_name: str = "meta-llama/Llama-3.2-1B"
    max_length: int = 128
    dtype: str = "torch.bfloat16"

    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    num_epochs: int = 1
    optim: str = "adamw_torch"

    # LoRA setup
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    lora_target_modules: str = "all-linear"

    # Other settings
    seed: int = 23
    output_dir: str = "experiments/results/llama32-1b-bs32-ft32-ml128-r4-a8-adamw_torch"
    wandb_project: str = "llama-finetuning"
    logging_steps: int = 10
    save_total_limit: int = 3

    def save(self, path: Optional[str] = None):
        """Save configuration to a JSON file."""
        if path is None:
            path = os.path.join(self.output_dir, "config.json")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, path: str):
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class SSTDataset:
    def __init__(self, config: TrainingConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.required_columns = ["input_ids", "attention_mask", "labels", "text", "label"]

    def _apply_template(self, example: Dict[str, Any], mode: str = "train") -> Dict[str, Any]:
        """Apply prompt template to dataset examples."""
        if mode == "train":
            example["text"] = TRAIN_PROMPT_TEMPLATE.substitute(
                input=example["sentence"], label=LBL2VALUE[example["label"]]
            )
        else:
            example["text"] = TEST_PROMPT_TEMPLATE.substitute(input=example["sentence"])
        return example

    def _tokenize_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize input text and labels."""
        tokenized_batch = self.tokenizer(
            example["text"], padding="max_length", max_length=self.config.max_length, truncation=True
        )

        # Convert labels to strings
        example["label"] = [LBL2VALUE[lbl] for lbl in example["label"]]
        tokenized_lbls = self.tokenizer(
            example["label"], padding="max_length", max_length=self.config.max_length, truncation=True
        )

        tokenized_batch["labels"] = tokenized_lbls["input_ids"]
        return tokenized_batch

    def load(self) -> Tuple[Any, Any]:
        """Load and preprocess dataset."""
        print(f"Loading dataset ({self.config.dataset_id})...")
        dataset = load_dataset(self.config.dataset_id)

        # Prepare training set
        train_set = dataset["train"].map(self._apply_template, fn_kwargs={"mode": "train"})
        tokenized_train_set = train_set.map(self._tokenize_function, batched=True)
        tokenized_train_set.set_format("torch", columns=self.required_columns)

        # Prepare evaluation set
        eval_set = dataset["validation"].map(self._apply_template, fn_kwargs={"mode": "test"})
        tokenized_eval_set = eval_set.map(self._tokenize_function, batched=True)
        tokenized_eval_set.set_format("torch", columns=self.required_columns)

        return tokenized_train_set, tokenized_eval_set
