# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel
import torch

from thomas.tooling.types import create_mapped_type
from thomas.models.torch.dtypes import TorchDType
from thomas.models.torch.loss import TorchLoss
from thomas.models.torch.opt import TorchOptimizer


def test_create_mapped_type():
    class Model(BaseModel):
        dtype: TorchDType
        loss: TorchLoss
        optimizer: TorchOptimizer

    assert Model(dtype="float32", loss="MSELoss", optimizer="Adam")
    assert Model(dtype=torch.float32, loss=torch.nn.MSELoss, optimizer=torch.optim.Adam)
    try:
        Model(dtype="float64", loss=1, optimizer="A")
        assert False
    except ValueError as e:
        print(e)
        assert True


if __name__ == "__main__":
    test_create_mapped_type()
