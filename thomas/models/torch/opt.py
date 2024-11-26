# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torch import optim

from ..types import create_mapped_type

# Maybe should be frozen or somehow protected
map_optimizer = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
}

# Create the Annotated types
TorchOptimizer = create_mapped_type(map_optimizer)
