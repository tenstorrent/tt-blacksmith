# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.torch_helpers import collate_fn_for_causal_lm
from blacksmith.tools.trainer.trainer import Trainer


class LoraLLMTrainer(Trainer):
    """
    Trainer for parameter-efficient fine-tuning of causal LLMs using LoRA.
    """

    def _load_model(self) -> torch.nn.Module:
        return get_model(self.config, self.device_manager.device, compile_model=True)

    def _load_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_dataset = get_dataset(config=self.config, split="train", collate_fn=collate_fn_for_causal_lm)
        val_dataset = get_dataset(config=self.config, split="validation", collate_fn=collate_fn_for_causal_lm)

        return train_dataset.get_dataloader(), val_dataset.get_dataloader()

    def _forward(self, batch: Any) -> torch.Tensor:
        output = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        # Shift logits for causal LM: position t predicts token t+1. Labels are
        # already shifted in the collate fn, so we drop the final logit here.
        shift_logits = output.logits[:, :-1, :].contiguous()
        labels = batch["labels"]

        return F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.config.ignored_index,
        )
