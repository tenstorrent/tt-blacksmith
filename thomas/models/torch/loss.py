# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union, Literal

from torch import nn
from pydantic import BaseModel

from thomas.tooling.types import create_mapped_type

# Maybe should be frozen or somehow protected
map_loss = {
    "MSELoss": nn.MSELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
}

# Create the Annotated types
TorchLoss = create_mapped_type(map_loss, Union[Literal["MSELoss"], Literal["CrossEntropyLoss"]])

if __name__ == "__main__":
    class Model(BaseModel):
        loss: TorchLoss
    
    model = Model(loss="MSELoss")
    print(model)
    model = Model(loss="CrossEntropyLoss")
    print(model)
    model = Model(loss=nn.CrossEntropyLoss)
    print(model)
    try:
        model = Model(loss="BCELoss")
    except ValueError as e:
        print(e)
    try:
        model = Model(loss=nn.BCELoss)
    except ValueError as e:
        print(e)
    try:
        model = Model(loss=5)
    except ValueError as e:
        print(e)