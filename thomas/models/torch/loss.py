# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from torch import nn

from ..types import create_mapped_type
from forge.op.loss import CrossEntropyLoss, L1Loss

# Maybe should be frozen or somehow protected
map_loss = {
    "L1Loss": L1Loss,
    "CrossEntropyLoss": CrossEntropyLoss,
}

# Create the Annotated types
TorchLoss = create_mapped_type(map_loss)
