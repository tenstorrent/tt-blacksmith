# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import optax
from typing import Any, Tuple, Optional, List


def radam(
    learning_rate: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    degenerated_to_sgd: bool = True,
) -> optax.GradientTransformation:
    """
    Rectified Adam optimizer implemented in JAX/Optax, matching PyTorch RAdam.

    Args:
        learning_rate: Learning rate.
        betas: Coefficients (beta1, beta2) for computing running averages of gradient and its square.
        eps: Term added to denominator for numerical stability.
        weight_decay: Weight decay (L2 penalty).
        degenerated_to_sgd: Whether to degenerate to SGD when N_sma < 5.

    Returns:
        optax.GradientTransformation: An Optax optimizer.
    """
    if not 0.0 <= learning_rate:
        raise ValueError(f"Invalid learning rate: {learning_rate}")
    if not 0.0 <= eps:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

    def init_fn(params: Any) -> optax.OptState:
        return {
            "step": jnp.zeros([], dtype=jnp.int32),
            "exp_avg": jax.tree_map(jnp.zeros_like, params),
            "exp_avg_sq": jax.tree_map(jnp.zeros_like, params),
            "buffer": [jnp.array([0, 0.0, 0.0], dtype=jnp.float32) for _ in range(10)],
        }

    def update_fn(updates: Any, state: optax.OptState, params: Optional[Any] = None) -> Tuple[Any, optax.OptState]:
        beta1, beta2 = betas
        step = state["step"] + 1

        exp_avg = jax.tree_map(lambda ea, g: ea * beta1 + (1 - beta1) * g, state["exp_avg"], updates)
        exp_avg_sq = jax.tree_map(lambda eas, g: eas * beta2 + (1 - beta2) * g * g, state["exp_avg_sq"], updates)

        buffer_idx = step % 10
        buffered = state["buffer"][buffer_idx]

        def compute_step_size(step_idx):
            beta2_t = beta2**step_idx
            N_sma_max = 2 / (1 - beta2) - 1
            N_sma = N_sma_max - 2 * step_idx * beta2_t / (1 - beta2_t)
            N_sma = jnp.maximum(N_sma, 0.0)  # Match PyTorch’s conservative approach

            def radam_step():
                rectification = jnp.sqrt(
                    (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)
                )
                return rectification / (1 - beta1**step_idx)

            def sgd_step():
                return 1.0 / (1 - beta1**step_idx)

            step_size = jax.lax.cond(
                N_sma >= 5, radam_step, lambda: sgd_step() if degenerated_to_sgd else -1.0  # Match PyTorch’s -1
            )
            return N_sma, step_size

        N_sma, step_size = jax.lax.cond(
            step == buffered[0], lambda: (buffered[1], buffered[2]), lambda: compute_step_size(step)
        )
        new_buffer = [
            buf if i != buffer_idx else jnp.array([step, N_sma, step_size]) for i, buf in enumerate(state["buffer"])
        ]

        def radam_update(p, g, ea, eas):
            p_new = p - weight_decay * learning_rate * p if weight_decay != 0 else p
            denom = jnp.sqrt(eas) + eps
            return p_new - step_size * learning_rate * ea / denom

        def sgd_update(p, g, ea):
            p_new = p - weight_decay * learning_rate * p if weight_decay != 0 else p
            return p_new - step_size * learning_rate * ea

        updates = jax.tree_map(
            lambda p, g, ea, eas: jax.lax.cond(
                N_sma >= 5,
                lambda: radam_update(p, g, ea, eas),
                lambda: jax.lax.cond(step_size > 0, lambda: sgd_update(p, g, ea), lambda: p),
            ),
            params,
            updates,
            exp_avg,
            exp_avg_sq,
        )

        new_state = {"step": step, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq, "buffer": new_buffer}
        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def get_optimizer(config, models: List[Any]) -> optax.GradientTransformation:
    """
    Create an optimizer based on config for the given models.

    Args:
        config: NerfConfig object with training.optimizer and other settings
        models: List of models (unused here, kept for compatibility)
    Returns:
        optax.GradientTransformation: Configured optimizer
    """
    optimizer_kwargs = config.training.optimizer_kwargs or {}
    if config.training.optimizer == "radam":
        return optax.radam(
            learning_rate=optimizer_kwargs.get("lr", 8e-4),
            b1=optimizer_kwargs.get("betas", (0.9, 0.999))[0],  # Extract beta1
            b2=optimizer_kwargs.get("betas", (0.9, 0.999))[1],  # Extract beta2
            eps=optimizer_kwargs.get("eps", 1e-8),
        )
    elif config.training.optimizer == "adam":
        return optax.adam(learning_rate=optimizer_kwargs.get("lr", 5e-4), b1=0.9, b2=0.999, eps=1e-8)
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
