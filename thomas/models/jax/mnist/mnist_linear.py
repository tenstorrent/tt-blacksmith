# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from thomas.models.jax.mnist.net_config import NetConfig


class MNISTLinearConfig(NetConfig):
    input_size: int = 784
    output_size: int = 10
    hidden_size: int = 128
