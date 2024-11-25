# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Union

import forge
import torch
from torch import nn
import lightning as L

from thomas.models.torch.loss import TorchLoss


@dataclass
class LightningConfig:
    batch_size: int
    input_size: int
    loss: TorchLoss


class TTLightningModel(L.LightningModule):
    def __init__(self, config: LightningConfig, model: nn.Module):
        super(TTLightningModel, self).__init__()
        self.save_hyperparameters()
        self.framework_model = model
        tt_model = forge.compile(
            self.framework_model,
            sample_inputs=[torch.rand(config.batch_size, config.input_size)],
            loss=config.loss(),
        )
        self.model = tt_model
        self.loss = config.loss()

    def forward(self, x):
        logits = self.model(x)
        logits = logits[0]
        return logits

    def backward(self, loss, *args, **kwargs):
        loss.backward(*args, **kwargs)
        self.model.backward()

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        acc = (pred.argmax(1) == y).type(torch.float).mean()
        self.log("val/loss", loss)
        self.log("val/acc", acc)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)
