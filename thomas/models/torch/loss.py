# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, field_validator, validator
from torch import nn

from ..types import create_mapped_type
import forge.op.loss as forge_loss
from torch.nn import L1Loss, CrossEntropyLoss

# Maybe should be frozen or somehow protected
map_loss = {
    "L1Loss": {"cpu": L1Loss, "tt": forge_loss.L1Loss},
    "CrossEntropyLoss": {"cpu": CrossEntropyLoss, "tt": forge_loss.CrossEntropyLoss},
}

# each loss should have both cpu and tt implementations
assert all({"cpu", "tt"} == set(v.keys()) for v in map_loss.values())

TorchLoss = create_mapped_type({k: v["cpu"] for k, v in map_loss.items()})
ForgeLoss = create_mapped_type({k: v["tt"] for k, v in map_loss.items()})


class LossConfig(BaseModel):
    run_on: str = "cpu"
    loss_name: str

    @field_validator("loss_name")
    def validate_loss_name(cls, v):
        if v not in map_loss:
            raise ValueError(f"Invalid loss name: {v}")
        return v

    @property
    def loss(self):
        if self.run_on == "cpu":
            return map_loss[self.loss_name]["cpu"]
        else:
            return map_loss[self.loss_name]["tt"]
