# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    batch_size: int
    epochs: int
    lr: float
