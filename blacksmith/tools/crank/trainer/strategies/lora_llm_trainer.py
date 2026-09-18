# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch

from blacksmith.models.torch.huggingface.hf_models import get_model
from blacksmith.tools.crank.trainer.trainer import Trainer
from blacksmith.tools.trainer.strategies.lora_llm_trainer import (
    LoraLLMTrainer as XlaLoraLLMTrainer,
)
from blacksmith.tools.trainer.strategies.lora_llm_trainer import compute_causal_lm_loss
from blacksmith.tools.workaround_utils import cross_entropy_loss


class LoraLLMTrainer(Trainer, XlaLoraLLMTrainer):
    """
    Trainer for parameter-efficient fine-tuning of causal LLMs using LoRA, on tt-crank.

    Data loading, the loss probe and the label transform come from the tt-xla strategy;
    only how the model is compiled and how a batch is run differ.
    """

    def _load_model(self) -> torch.nn.Module:
        # compile_model=False: forward + loss are compiled as one callable so the loss
        # and its backward stay inside the tt graph.
        model = get_model(self.config, self.device_manager.device, compile_model=False)
        self.eval_model = model
        self._compute_loss_fn = compute_causal_lm_loss
        if self.config.use_tt:
            # dynamic=False: tt-crank cannot lower symbolic shapes, so never let dynamo
            # generalize a recompile into SymInts.
            self._compute_loss_fn = torch.compile(
                compute_causal_lm_loss,
                backend="tt",
                dynamic=False,
                options=self.device_manager.compile_options(),
            )
        return model

    def _forward(self, batch: Any) -> torch.Tensor:
        # One-hot on CPU (labels were left on the host), then move / shard the targets.
        # Validation goes through the same compiled forward+loss as training: an eager
        # loss over the logits is wrong on a mesh (see llama `train_crank.py`), and on a
        # single chip the numbers are identical.
        batch = self.device_manager.prepare_batch(self._labels_to_targets(batch))
        return self._compute_loss_fn(batch, self.model, cross_entropy_loss)
