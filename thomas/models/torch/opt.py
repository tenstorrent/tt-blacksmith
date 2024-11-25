# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Union, Literal

from torch import optim

from thomas.tooling.types import create_mapped_type

map_optimizer = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
}

# Create the Annotated types
TorchOptimizer = create_mapped_type(map_optimizer, Union[Literal["SGD"], Literal["Adam"], Literal["AdamW"]])