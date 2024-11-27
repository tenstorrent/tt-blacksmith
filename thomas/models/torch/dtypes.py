# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

from ..types import create_mapped_type

# Maybe should be frozen or somehow protected
torch_map_dtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

# Create the Annotated types
TorchDType = create_mapped_type(torch_map_dtype)
