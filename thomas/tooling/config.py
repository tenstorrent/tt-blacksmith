# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel
from typing import List
from thomas.models.torch.dtypes import TorchDType


class DataLoadingConfig(BaseModel):
    batch_size: int
    dtype: TorchDType
    pre_shuffle: bool
