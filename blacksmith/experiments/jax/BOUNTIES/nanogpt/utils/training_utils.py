# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import optax
from typing import Any, Tuple, Callable, Optional
import flax
from flax import struct
import math
import logging

logger = logging.getLogger(__name__)


@struct.dataclass
class TrainState:
    """Training state containing model parameters and optimizer state."""
    step: int
    params: Any
    opt_state: Any
    model: Any = struct.field(pytree_node=False)
    optimizer: Any = struct.field(pytree_node=False)


def create_optimizer(config) -> optax.GradientTransformation:
    """Create optimizer from configuration."""
    # Learning rate schedule
    def lr_schedule(step: int) -> float:
        if step < config.training.warmup_iters:
            # Linear warmup
            return config.training.learning_rate * step / config.training.warmup_iters
        elif step > config.training.lr_decay_iters:
            # Linear decay
            return config.training.min_lr
        else:
            # Cosine decay
            decay_ratio = (step - config.training.warmup_iters) / (config.training.lr_decay_iters - config.training.warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return config.training.min_lr + coeff * (config.training.learning_rate - config.training.min_lr)
    
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.training.grad_clip),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.training.beta1,
            b2=config.training.beta2,
            weight_decay=config.training.weight_decay,
            eps=1e-8
        )
    )
    
    return optimizer


def compute_loss(
    model: Any,
    params: Any,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    training: bool = True
) -> Tuple[float, Any]:
    """Compute cross-entropy loss for language modeling."""
    
    # Forward pass
    logits = model.apply(params, inputs, training=training)
    
    # Compute loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    
    return loss, logits


def compute_loss_and_grads(
    model: Any,
    params: Any,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    training: bool = True
) -> Tuple[float, Any, Any]:
    """Compute loss and gradients."""
    
    def loss_fn(p):
        return compute_loss(model, p, inputs, targets, training)
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    return loss, grads, logits


def update_params(
    optimizer: optax.GradientTransformation,
    params: Any,
    opt_state: Any,
    grads: Any
) -> Tuple[Any, Any]:
    """Update model parameters using optimizer."""
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state


def estimate_loss(
    model: Any,
    params: Any,
    get_batch_fn: Callable,
    eval_iters: int,
    device_manager: Any
) -> float:
    """Estimate validation loss."""
    
    losses = []
    
    for _ in range(eval_iters):
        # Get batch on primary device
        with device_manager.with_device(device_manager.primary_device):
            inputs, targets = get_batch_fn()
        
        # Compute loss
        loss, _ = compute_loss(model, params, inputs, targets, training=False)
        losses.append(loss)
    
    return jnp.mean(jnp.array(losses))


def get_lr(step: int, config) -> float:
    """Get current learning rate."""
    if step < config.training.warmup_iters:
        return config.training.learning_rate * step / config.training.warmup_iters
    elif step > config.training.lr_decay_iters:
        return config.training.min_lr
    else:
        decay_ratio = (step - config.training.warmup_iters) / (config.training.lr_decay_iters - config.training.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.training.min_lr + coeff * (config.training.learning_rate - config.training.min_lr)


def create_train_state(
    model: Any,
    params: Any,
    optimizer: optax.GradientTransformation,
    device_manager: Any
) -> TrainState:
    """Create initial training state with explicit CPU fallback for optimizer init."""
    
    logger.info("Creating initial training state")
    
    # Initialize optimizer state on CPU (explicit fallback)
    opt_state = device_manager.cpu_fallback(optimizer.init, params)
    
    train_state = TrainState(
        step=0,
        params=params,
        opt_state=opt_state,
        model=model,
        optimizer=optimizer
    )
    logger.info("Training state created successfully")
    return train_state


def training_step(
    train_state: TrainState,
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
    device_manager: Any
) -> Tuple[TrainState, float, Any]:
    """Perform a single training step."""
    
    # Compute loss and gradients on primary device
    with device_manager.with_device(device_manager.primary_device):
        loss, grads, logits = compute_loss_and_grads(
            train_state.model,
            train_state.params,
            inputs,
            targets,
            training=True
        )
    
    # Update parameters
    new_params, new_opt_state = update_params(
        train_state.optimizer,
        train_state.params,
        train_state.opt_state,
        grads
    )
    
    # Update training state
    new_train_state = train_state.replace(
        step=train_state.step + 1,
        params=new_params,
        opt_state=new_opt_state
    )
    
    return new_train_state, loss, logits


def save_checkpoint(
    train_state: TrainState,
    checkpoint_path: str,
    step: int
) -> None:
    """Save model checkpoint."""
    
    checkpoint_data = {
        'step': step,
        'params': train_state.params,
        'opt_state': train_state.opt_state
    }
    
    # Save using JAX checkpoint format
    with open(checkpoint_path, 'wb') as f:
        import pickle
        pickle.dump(checkpoint_data, f)


def load_checkpoint(
    checkpoint_path: str,
    train_state: TrainState
) -> TrainState:
    """Load model checkpoint."""
    
    with open(checkpoint_path, 'rb') as f:
        import pickle
        checkpoint_data = pickle.load(f)
    
    return train_state.replace(
        step=checkpoint_data['step'],
        params=checkpoint_data['params'],
        opt_state=checkpoint_data['opt_state']
    )
