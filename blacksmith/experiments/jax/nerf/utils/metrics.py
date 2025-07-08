# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax.numpy as jnp


def mse(
    image_pred: jnp.ndarray, image_gt: jnp.ndarray, valid_mask: jnp.ndarray = None, reduction: str = "mean"
) -> jnp.ndarray:
    """
    Compute Mean Squared Error (MSE) between predicted and ground truth images in JAX.

    Args:
        image_pred: Predicted image array
        image_gt: Ground truth image array
        valid_mask: Optional boolean mask to select valid pixels
        reduction: Reduction method ("mean" or None for per-pixel values)
    Returns:
        MSE value (scalar if reduction="mean", array otherwise)
    """
    if valid_mask is not None:
        image_pred = image_pred[valid_mask]
        image_gt = image_gt[valid_mask]
    value = (image_pred - image_gt) ** 2
    if reduction == "mean":
        return jnp.mean(value)
    return value


def psnr(
    image_pred: jnp.ndarray, image_gt: jnp.ndarray, valid_mask: jnp.ndarray = None, reduction: str = "mean"
) -> jnp.ndarray:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between predicted and ground truth images in JAX.

    Args:
        image_pred: Predicted image array
        image_gt: Ground truth image array
        valid_mask: Optional boolean mask to select valid pixels
        reduction: Reduction method ("mean" or None for per-pixel values)
    Returns:
        PSNR value (scalar if reduction="mean", array otherwise)
    """
    mse_value = mse(image_pred, image_gt, valid_mask, reduction)
    return -10 * jnp.log10(mse_value)
