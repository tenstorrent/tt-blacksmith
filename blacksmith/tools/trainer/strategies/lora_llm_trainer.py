# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
from torch.utils.data import DataLoader

from blacksmith.datasets.torch.dataset_utils import get_dataset
from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.torch_helpers import collate_fn_for_causal_lm
from blacksmith.tools.trainer.trainer import Trainer
from blacksmith.tools.workaround_utils import cross_entropy_loss, transform_labels

TT_COMPILE_OPTIONS = {
    "tt_enable_torch_fx_fusion_pass": False,
    "tt_legacy_compile": True,
    "tt_lazy_execution": True,
    "tt_use_aot_autograd": False,
}
IGNORED_INDEX = -100


def compute_causal_lm_loss(batch, model, loss_fn):
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    shift_logits = output.logits[:, :-1, :].contiguous()
    return loss_fn(shift_logits, batch["expected_output"], batch["labels_mask"])


class LoraLLMTrainer(Trainer):
    """
    Trainer for parameter-efficient fine-tuning of causal LLMs using LoRA.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Defaults so subclasses that override `_load_model` (e.g. inference-
        # server JobLoraTrainer) still have a loss callable before train().
        self.eval_model = None
        self._compute_loss_fn = compute_causal_lm_loss

    def _load_model(self) -> torch.nn.Module:
        # compile_model=False: training goes through a compiled loss wrapper
        # (see llama experiment) so fwd+bwd can join under lazy execution.
        model = get_model(self.config, self.device_manager.device, compile_model=False)
        self.eval_model = model
        self._compute_loss_fn = compute_causal_lm_loss
        if self.config.use_tt:
            self._compute_loss_fn = torch.compile(
                compute_causal_lm_loss,
                backend="tt",
                options=TT_COMPILE_OPTIONS,
            )
            if self.config.val_steps_freq > 0:
                self.eval_model = torch.compile(model, backend="tt", options=TT_COMPILE_OPTIONS)
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

    def _make_step_loss(self) -> torch.Tensor:
        with torch.no_grad():
            loss_probe = cross_entropy_loss(torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.zeros(1, 1))
        return torch.zeros(
            loss_probe.shape,
            dtype=loss_probe.dtype,
            device=self.device_manager.device,
        )

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
        # One-hot on CPU (labels were left on the host). Train then copies the
        # targets to device like the llama experiment; val keeps them on CPU.
        batch = self._labels_to_targets(batch)
        if self.model.training:
            batch = self.device_manager.prepare_batch(batch)
            return self._compute_loss_fn(batch, self.model, cross_entropy_loss)
        if not self.config.use_tt:
            batch = self.device_manager.prepare_batch(batch)
        eval_model = getattr(self, "eval_model", self.model)
        return compute_causal_lm_loss(batch, eval_model, cross_entropy_loss)
