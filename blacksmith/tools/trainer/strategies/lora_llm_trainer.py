# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
from torch.utils.data import DataLoader

from blacksmith.datasets.dataset_utils import get_dataset
from blacksmith.models.hf_models import get_model
from blacksmith.tools.torch_helpers import collate_fn_for_causal_lm
from blacksmith.tools.trainer.trainer import Trainer
from blacksmith.tools.workaround_utils import cross_entropy_loss, transform_labels

IGNORED_INDEX = -100


def compute_causal_lm_loss(batch, model, loss_fn):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    shift_logits = output.logits[:, :-1, :].contiguous()
    return loss_fn(shift_logits, batch["expected_output"], batch["labels_mask"])


class LoraLLMTrainer(Trainer):
    """
    Trainer for parameter-efficient fine-tuning of causal LLMs using LoRA, on tt-crank.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Defaults so subclasses that override `_load_model` still have a loss
        # callable before train().
        self.eval_model = None
        self._compute_loss_fn = compute_causal_lm_loss

    def _load_model(self) -> torch.nn.Module:
        # The model is returned eager; forward + loss are compiled as one callable so the
        # loss and its backward stay inside the tt graph.
        model = get_model(self.config, self.device_manager.device)
        self.eval_model = model
        self._compute_loss_fn = compute_causal_lm_loss
        if self.config.use_tt:
            # dynamic=False: tt-crank cannot lower symbolic shapes.
            self._compute_loss_fn = torch.compile(
                compute_causal_lm_loss,
                backend="tt",
                dynamic=False,
                options=self.device_manager.compile_options(),
            )
        return model

    def _load_dataloaders(self) -> tuple[DataLoader, DataLoader | None]:
        train_dataset = get_dataset(config=self.config, split="train", collate_fn=collate_fn_for_causal_lm)
        if self.config.val_steps_freq == 0:
            return train_dataset.get_dataloader(), None
        val_dataset = get_dataset(
            config=self.config,
            split="validation",
            collate_fn=collate_fn_for_causal_lm,
        )
        return train_dataset.get_dataloader(), val_dataset.get_dataloader()

    def _labels_to_targets(self, batch: dict) -> dict:
        expected_output, labels_mask = transform_labels(
            batch["labels"],
            IGNORED_INDEX,
            self.model.config.vocab_size,
        )
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "expected_output": expected_output,
            "labels_mask": labels_mask,
        }

    def _forward(self, batch: Any) -> torch.Tensor:
        # One-hot on CPU (labels were left on the host), then move / shard the targets.
        batch = self.device_manager.prepare_batch(self._labels_to_targets(batch))
        return self._compute_loss_fn(batch, self.model, cross_entropy_loss)
