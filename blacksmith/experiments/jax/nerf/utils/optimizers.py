# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax


def get_optimizer(config, models: List[Any]) -> optax.GradientTransformation:
    if config.training.optimizer == "radam":
        return optax.radam(
            learning_rate=config.training.lr,
            b1=config.training.betas[0],
            b2=config.training.betas[1],
            eps=config.training.eps,
        )
    elif config.training.optimizer == "adam":
        return optax.adam(
            learning_rate=config.training.lr,
            b1=config.training.betas[0],
            b2=config.training.betas[1],
            eps=config.training.eps,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
