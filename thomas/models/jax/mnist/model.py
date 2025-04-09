# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import flax.linen as nn


class MLP(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.Dense(features=10)(x)
        return x


class Models:
    def __init__(self, model_type="MLP", hidden_size=128):
        if model_type == "MLP":
            self.model = MLP(hidden_size=hidden_size)
        # Add other model types here
        # elif model_type == 'CNN':
        #     self.model = CNN()
        # elif model_type == 'LLM':
        #     self.model = LLM()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def __call__(self, x):
        return self.model(x)
