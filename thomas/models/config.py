# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel


class ModelConfig(BaseModel):
    input_size: int
    output_size: int
    hidden_size: int
    bias: bool
