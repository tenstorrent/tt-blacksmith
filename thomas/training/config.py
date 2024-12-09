# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel
from thomas.models.torch.loss import TorchLoss


class TrainingConfiguration(BaseModel):
    batch_size: int
    loss: TorchLoss
    epochs: int
    lr: float
