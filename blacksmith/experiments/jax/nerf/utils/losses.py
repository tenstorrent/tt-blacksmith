# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict

import jax
import jax.numpy as jnp
from flax import linen as nn


class MSELoss(nn.Module):
    """Mean Squared Error Loss implemented in Flax."""

    @nn.compact
    def __call__(self, inputs: Dict[str, jnp.ndarray], targets: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            inputs: Dictionary containing 'rgb_coarse' and/or 'rgb_fine' predictions.
            targets: Ground truth RGB values.
        Returns:
            Scalar loss value.
        """

        def mse_loss(pred, target):
            return jnp.mean(jnp.square(pred - target))

        loss = 0.0
        if "rgb_coarse" in inputs:
            loss += mse_loss(inputs["rgb_coarse"], targets)
        if "rgb_fine" in inputs:
            loss += mse_loss(inputs["rgb_fine"], targets)

        return loss


loss_dict = {"mse": MSELoss}
