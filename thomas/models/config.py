# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel
from thomas.models.torch.dtypes import TorchDType


class NetConfig(BaseModel):
    input_size: int
    output_size: int


class MNISTLinearConfig(NetConfig):
    batch_size: int
    hidden_size: int
    bias: bool = True
    # Should be decides if it is Torch or for example Jax dtype
    dtype: TorchDType = "float32"
