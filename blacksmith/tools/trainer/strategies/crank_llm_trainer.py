# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from blacksmith.configs.checkpoint import CheckpointConfig
from blacksmith.datasets.dataset_utils import get_dataset
from blacksmith.datasets.llm_dataset import LLMDataset
from blacksmith.models.hf_models import get_model
from blacksmith.tools.crank.checkpoint_manager import CrankCheckpointManager
from blacksmith.tools.crank.device_manager import CrankDeviceManager
from blacksmith.tools.crank.reproducibility_manager import CrankReproducibilityManager
from blacksmith.tools.crank.torch_helpers import (
    collate_fn_for_causal_lm,
    loss_to_float,
    to_host,
    tt_compile_options,
)
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.trainer.trainer import Trainer

IGNORED_INDEX = -100


def compute_causal_lm_loss(batch, model):
    # No padding mask: batches are right-padded and padded labels are IGNORED_INDEX, so causal attention is
    # exact for every real token, and without a mask tt-crank's sdpa stays on its fused kernels at any batch size.
    output = model(input_ids=batch["input_ids"])
    # Labels are pre-shifted by collate_fn_for_causal_lm, so logits[:, :-1] line up with them.
    shift_logits = output.logits[:, :-1, :]
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), batch["labels"].reshape(-1), ignore_index=IGNORED_INDEX
    )


class CrankLLMTrainer(Trainer):
    """
    Trainer for fine-tuning causal LLMs on tt-crank (torch backend).

    LoRA is opt-in via the nested ``lora`` config block; without it the trainer
    does a full fine-tune.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default so subclasses that override `_load_model` still have a loss
        # callable before train().
        self._compute_loss_fn = compute_causal_lm_loss

    # --------------------------------------------------------- manager factories
    def _create_reproducibility_manager(self) -> CrankReproducibilityManager:
        return CrankReproducibilityManager(self.config)

    def _create_device_manager(self) -> CrankDeviceManager:
        return CrankDeviceManager(self.config)

    def create_checkpoint_manager(
        self,
        checkpoint_config: CheckpointConfig,
        logger: TrainingLogger | None = None,
    ) -> CrankCheckpointManager:
        return CrankCheckpointManager(checkpoint_config, logger=logger)

    # ------------------------------------------------------------------- hooks
    def _load_model(self) -> torch.nn.Module:
        # The model is returned eager; forward + loss are compiled as one callable so the
        # loss and its backward stay inside the tt graph.
        model = get_model(self.config, self.device_manager.device)
        self._compute_loss_fn = compute_causal_lm_loss
        if self.config.use_tt:
            # dynamic=False: tt-crank cannot lower symbolic shapes.
            self._compute_loss_fn = torch.compile(
                compute_causal_lm_loss,
                backend="tt",
                dynamic=False,
                options=tt_compile_options(self.config),
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

    def _load_optimizer(self) -> torch.optim.Optimizer:
        """Build an AdamW optimizer over the model's trainable parameters."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        # fused=True: tt-crank implements `_fused_adamw_` natively, while torch's default
        # (foreach) AdamW runs as a chain of small eager ops per parameter on tt.
        return torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            fused=True,
        )

    def _forward(self, batch: Any) -> torch.Tensor:
        return self._compute_loss_fn(batch, self.model)

    def _backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def _optimizer_step(self) -> None:
        self.optimizer.step()
        self._zero_grad()

    def _zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def _train_mode(self) -> None:
        self.model.train()

    def _eval_mode(self) -> None:
        self.model.eval()

    def _no_grad(self):
        return torch.no_grad()

    def _accumulate_loss(self, running, loss: torch.Tensor) -> torch.Tensor:
        # Detach so the autograd graph is not retained; the add stays on the
        # device (no host readback), so the compiled training step is not broken.
        detached = loss.detach()
        return detached if running is None else running + detached

    def loss_to_float(self, loss: torch.Tensor | float) -> float:
        return loss_to_float(loss)

    def parameter_counts(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def current_learning_rate(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def save_adapter(self, path: str) -> None:
        if self.config.lora is None:
            raise ValueError("save_adapter needs a LoRA run; this run fine-tunes the full model")
        # LoRA weights only, on the host (gathers DTensor shards; `tt` tensors can't be serialized in place).
        trainable = {name for name, param in self.model.named_parameters() if param.requires_grad}
        state_dict = to_host({name: v for name, v in self.model.state_dict().items() if name in trainable})
        self.model.save_pretrained(path, state_dict=state_dict)

    def cleanup(self) -> None:
        # The tokenized-dataset cache is process-wide; the next run may use another model / length.
        LLMDataset.clear_cache()
        super().cleanup()
