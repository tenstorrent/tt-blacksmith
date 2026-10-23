# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np
import ttml
import ttml.common.utils
import ttnn

from blacksmith.configs.checkpoint import CheckpointConfig
from blacksmith.datasets.dataset_utils import get_dataset
from blacksmith.datasets.llm_dataset import LLMDataset
from blacksmith.models.ttml import get_model, get_provider
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.trainer.trainer import Trainer
from blacksmith.tools.ttml.checkpoint_manager import TtmlCheckpointManager
from blacksmith.tools.ttml.device_manager import TtmlDeviceManager
from blacksmith.tools.ttml.reproducibility_manager import TtmlReproducibilityManager
from blacksmith.tools.ttml.ttml_helpers import build_dataloader
from blacksmith.tools.ttml.utilities import write_adapter


class TtmlLLMTrainer(Trainer):
    """
    Trainer for LoRA fine-tuning of causal LLMs on ttml (tt-train), single chip.

    The model comes from a ttml model provider (``blacksmith.models.ttml``), which also decides
    whether the config can be trained at all; this trainer has no config checks of its own.
    Losses are read back to host floats once per micro-batch, as ttml's own trainers do: there
    is no compiled graph to protect.
    """

    # --------------------------------------------------------- manager factories
    def _create_reproducibility_manager(self) -> TtmlReproducibilityManager:
        return TtmlReproducibilityManager(self.config)

    def _create_device_manager(self) -> TtmlDeviceManager:
        return TtmlDeviceManager(self.config)

    def create_checkpoint_manager(
        self,
        checkpoint_config: CheckpointConfig,
        logger: TrainingLogger | None = None,
    ) -> TtmlCheckpointManager:
        # Checkpoints are PEFT adapters; the model provider maps them to and from the model.
        return TtmlCheckpointManager(checkpoint_config, self.provider, self.config, logger=logger)

    # ------------------------------------------------------------------- hooks
    def _load_model(self) -> Any:
        self.provider, _ = get_provider(self.config)
        return get_model(self.config)

    def _load_dataloaders(self) -> tuple[Any, Any | None]:
        train_dataset = get_dataset(config=self.config, split="train")
        if self.config.val_steps_freq == 0:
            return build_dataloader(train_dataset, self.config), None
        val_dataset = get_dataset(config=self.config, split="validation")
        return build_dataloader(train_dataset, self.config), build_dataloader(val_dataset, self.config)

    def _load_optimizer(self) -> Any:
        """AdamW over the trainable (LoRA) parameters, at a constant learning rate."""
        trainable = ttml.NamedParameters()
        for name, tensor in self.model.parameters().items():
            if tensor.get_requires_grad():
                trainable[name] = tensor
        return ttml.optimizers.create_optimizer(
            {"type": "AdamW", "lr": self.config.learning_rate, "weight_decay": self.config.weight_decay},
            trainable,
        )

    def _forward(self, batch: Any) -> Any:
        # mask=None: causal attention on device. Batches are right-padded and padded positions have a
        # zero loss_mask, so causal attention is exact for every real token (as in crank).
        logits = self.model(batch["input_ids"], None)
        per_token = ttml.ops.loss.cross_entropy_loss(logits, batch["labels"], ttml.ops.ReduceType.NONE)
        # loss_mask is normalized to sum to B * T, so the mean is over real targets only.
        return ttml.ops.unary.mean(per_token * batch["loss_mask"])

    def _scale_loss(self, loss: Any, divisor: int) -> Any:
        # ttml tensors multiply by a float but have no true division.
        return loss if divisor == 1 else loss * (1.0 / divisor)

    def _backward(self, loss: Any) -> None:
        loss.backward(False)

    def _clean_graph_state(self) -> None:
        # Drop this batch's autograd graph; gradients stay on the parameters until zero_grad.
        ttml.autograd.AutoContext.get_instance().reset_graph()

    def _optimizer_step(self) -> None:
        self.optimizer.step()
        self._zero_grad()

    def _zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def _train_mode(self) -> None:
        self.model.train()

    def _eval_mode(self) -> None:
        self.model.eval()

    def _no_grad(self):
        return ttml.common.utils.no_grad()

    def _accumulate_loss(self, running: float | None, loss: Any) -> float:
        # One host readback per micro-batch, before `_clean_graph_state` drops the graph.
        value = float(loss.to_numpy(ttnn.DataType.FLOAT32).mean())
        return value if running is None else running + value

    def loss_to_float(self, loss: float) -> float:
        return float(loss)

    def parameter_counts(self) -> tuple[int, int]:
        tensors = list(self.model.parameters().values())
        total = sum(int(np.prod(t.shape())) for t in tensors)
        trainable = sum(int(np.prod(t.shape())) for t in tensors if t.get_requires_grad())
        return total, trainable

    def current_learning_rate(self) -> float:
        return self.optimizer.get_lr()

    def save_adapter(self, path: str) -> None:
        write_adapter(path, self.provider.export_adapter(self.model, self.config))

    def cleanup(self) -> None:
        # The tokenized-dataset cache is process-wide; the next run may use another model / length.
        LLMDataset.clear_cache()
        super().cleanup()
