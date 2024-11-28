# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel


# Consts
CONFIG_PATH = "thomas/training/pytorch_train/config.yaml"

# Config model
class TrainConfig(BaseModel):
    model_id: str
    dataset_id: str
    output_path: str
    device: str
    max_length: int
    lr: float
    num_epochs: int
    batch_size: int
    lora_r: int
    lora_alpha: int
    save_steps: int
