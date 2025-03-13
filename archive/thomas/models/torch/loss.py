# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, field_validator, validator
from torch import nn

from thomas.models.types import create_mapped_type, DeviceType, Device
import forge.op.loss as forge_loss
from torch.nn import L1Loss, CrossEntropyLoss

# Maybe should be frozen or somehow protected
map_loss = {
    "L1Loss": {Device.cpu: L1Loss, Device.tt: forge_loss.L1Loss},
    "CrossEntropyLoss": {Device.cpu: CrossEntropyLoss, Device.tt: forge_loss.CrossEntropyLoss},
}

# each loss should have both cpu and tt implementations
assert all({Device.cpu, Device.tt} == set(v.keys()) for v in map_loss.values())

TorchLoss = create_mapped_type({k: v[Device.cpu] for k, v in map_loss.items()})
ForgeLoss = create_mapped_type({k: v[Device.tt] for k, v in map_loss.items()})


class LossConfig(BaseModel):
    run_on: DeviceType = Device.cpu
    loss_name: str

    @field_validator("loss_name")
    def validate_loss_name(cls, v):
        if v not in map_loss:
            raise ValueError(f"Invalid loss name: {v}")
        return v

    @property
    def loss(self):
        return map_loss[self.loss_name][self.run_on]
