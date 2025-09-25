# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Portions of this file are derived from 'lorax' by davisyoshida (MIT).
Copyright (c) 2023 davisyoshida
Source: https://github.com/davisyoshida/lorax
See THIRD_PARTY_NOTICES.md for the full MIT license text.
"""
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map_with_path, DictKey, SequenceKey

import optax
import quax

from .constants import LORA_FREEZE, LORA_FULL
from .transform import LoraWeight
from typing import Any, Tuple, Dict


def init_lora(param_tree, spec, rng, stddev=0.01, dtype=jnp.float32, alpha=1.0, is_leaf=None) -> Any:
    def iter_keys(key):
        # Infinite PRNGKey generator: repeatedly split the key to obtain a fresh
        # subkey for initializing each LoRA adapter tensor.
        while True:
            key, out_key = jax.random.split(key)
            yield out_key

    key_it = iter_keys(rng)

    def get_param(path, param, spec_val):
        # Map a single parameter and spec value to either:
        # - the original parameter (freeze/full tune), or
        # - a LoraWeight wrapping the base weight with newly initialized A/B.
        if spec_val in (LORA_FREEZE, LORA_FULL):
            return param

        if len(param.shape) == 1:
            raise ValueError(
                f"Vectors must either be frozen or fully tuned, but got spec value {spec} for param with path {path}"
            )

        if len(param.shape) == 2:
            b_dim, a_dim = param.shape

            b = jnp.zeros((b_dim, spec_val), dtype=dtype)
            a = jax.random.normal(next(key_it), (spec_val, a_dim), dtype=dtype) * stddev
            return LoraWeight(w=param, a=a, b=b, alpha=alpha)

        # conv case
        *window_shape, in_channels, out_channels = param.shape

        a = jnp.zeros(
            (*(1 for _ in range(len(window_shape))), spec_val, out_channels),
            dtype=param.dtype,
        )
        b = jax.random.normal(rng, (*window_shape, in_channels, spec_val), dtype=param.dtype) * stddev
        return LoraWeight(param, a, b, alpha=alpha)

    return jax.tree_util.tree_map_with_path(get_param, param_tree, spec, is_leaf=is_leaf)


def simple_spec(params, decision_fn=None, tune_vectors=False, is_leaf=None) -> Any:
    """
    Create a simple lora spec for a pytree
    Args:
        params: pytree of parameters
        tune_vectors: If true, will flag all arrays with less than 2 dimensions for tuning
        decision_fn: A function which maps a Jax KeyPath and a parameter to a spec value
    """
    if decision_fn is None:

        def decision_fn(*args):
            return LORA_FREEZE

    def full_fn(path, arr):
        if len(arr.shape) < 2:
            return LORA_FULL if tune_vectors else LORA_FREEZE

        value = decision_fn(path, arr)
        return value

    return tree_map_with_path(full_fn, params, is_leaf=is_leaf)


# NOTE: Optional helper for single-tree workflows. Not used in the current
# split/merge training loop (where we keep separate trainable/frozen trees).
# Use this if you want to materialise LoRA into dense weights and operate on a
# single params tree without `LoraWeight` wrappers.
def merge_params(lora_params, destructive=True, use_scaling=True) -> Any:
    """
    Re-merge LoRA parameters.
    Arguments:
        destructive: If True, the buffers in frozen_params may be freed to save memory.
        use_scaling: Whether to multiply LoRA params by alpha/r
    """
    if not use_scaling:
        raise ValueError("Scaling is now always enabled to match the original LoRA implementation.")

    def map_fn(param):
        if isinstance(param, LoraWeight):
            result = param.materialise()
            # Skip destructive deletion for now to avoid JAX API issues
            return result
        return param

    # Use tree_map with is_leaf to handle LoraWeight objects properly
    return jax.tree.map(map_fn, lora_params, is_leaf=lambda x: isinstance(x, LoraWeight))


# NOTE: Optional helper for checkpointing adapters only. Not used in the
# current split/merge loop (we already have `trainable_params`). Use this to
# produce a LoRA-only checkpoint retaining the full path structure.
def split_lora_params(params, spec) -> Any:
    """
    Map params to a pytree in which all `LoraWeight.w` values and all params marked with
    LORA_FREEZE are replaced with None. This is useful for checkpointing just
    the trainable params.
    """

    def node_mapper(node, spec_val):
        if not isinstance(node, LoraWeight):
            return node if spec_val != LORA_FREEZE else None
        # Create a new LoraWeight with w=None to save memory during checkpointing
        return LoraWeight(w=None, a=node.a, b=node.b, alpha=node.alpha)

    return jax.tree.map(node_mapper, params, spec)


# NOTE: Optional helper for single-tree optimizer updates. Not used in the
# current design because we compute grads only w.r.t. trainable params and
# update only those. Use this if you keep a merged params tree and want the
# optimizer to zero-out updates on frozen parts according to the spec.
def wrap_optimizer(
    optimizer: optax.GradientTransformation, spec, scalar_frozen_grads=False
) -> optax.GradientTransformation:
    """
    Wrap the optimizer to freeze parameters according to the LoRA spec.
    - LoraWeight objects: freeze 'w' and 'alpha', allow 'a' and 'b' to update
    - Regular parameters with spec=0 (LORA_FREEZE): completely frozen
    - Regular parameters with spec!=0: trainable
    """

    def freeze_weights(updates):
        def freeze_by_spec(update, spec_value):
            if isinstance(update, LoraWeight):
                # For LoraWeight: freeze 'w' and 'alpha', allow 'a' and 'b' to update
                return LoraWeight(
                    w=jnp.zeros_like(update.w),  # Zero gradient for frozen w
                    a=update.a,  # Keep gradient for a
                    b=update.b,  # Keep gradient for b
                    alpha=0.0,  # Freeze alpha
                )
            else:
                # For regular parameters: freeze if spec says LORA_FREEZE (0)
                if spec_value == 0:  # LORA_FREEZE
                    return jnp.zeros_like(update)  # Zero out gradients completely
                else:
                    return update  # Keep updates for non-frozen regular params

        return jax.tree.map(freeze_by_spec, updates, spec, is_leaf=lambda x: isinstance(x, LoraWeight))

    # Chain the freezing with the original optimizer
    return optax.chain(
        optax.GradientTransformation(
            init=lambda params: {},
            update=lambda updates, state, params: (freeze_weights(updates), state),
        ),
        optimizer,
    )


def split_trainable_frozen(lora_params, lora_spec) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split LoRA parameters into trainable and frozen pytrees.
    Trainable: Only LoRA a,b matrices from LoraWeight objects
    Frozen: Everything else (original weights w, alpha, and regular frozen params)
    """
    trainable_params = {}
    frozen_params = {}

    def split_param(param, spec_value, path_parts):
        if isinstance(param, LoraWeight):
            # For LoraWeight: only a,b are trainable
            trainable_params[".".join(path_parts)] = {"a": param.a, "b": param.b}
            frozen_params[".".join(path_parts)] = {"w": param.w, "alpha": param.alpha}
        else:
            # Regular parameters go to frozen (they should all be spec=0)
            frozen_params[".".join(path_parts)] = param

    def traverse_tree(params_tree, spec_tree, path=[]):
        if isinstance(params_tree, dict):
            for key in params_tree:
                traverse_tree(params_tree[key], spec_tree[key], path + [key])
        else:
            split_param(params_tree, spec_tree, path)

    traverse_tree(lora_params, lora_spec)

    print(f"Split completed:")
    print(f" Trainable params: {len(trainable_params)} LoRA matrix pairs")
    print(f" Frozen params: {len(frozen_params)} weight groups")

    return trainable_params, frozen_params


def merge_trainable_frozen(trainable_params, frozen_params) -> Any:
    """
    Merge trainable and frozen pytrees back into full LoRA parameter tree.
    """
    merged_params = {}

    # First add all frozen regular parameters
    for path, param in frozen_params.items():
        if isinstance(param, dict) and "w" in param and "alpha" in param:
            # This is a frozen LoraWeight component - will be merged with trainable
            continue
        else:
            # Regular frozen parameter
            keys = path.split(".")
            current = merged_params
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = param

    # Now merge LoraWeight objects
    for path, trainable in trainable_params.items():
        frozen_lora = frozen_params[path]  # Should have 'w' and 'alpha'

        # Reconstruct LoraWeight
        lora_weight = LoraWeight(w=frozen_lora["w"], a=trainable["a"], b=trainable["b"], alpha=frozen_lora["alpha"])

        # Place in merged tree
        keys = path.split(".")
        current = merged_params
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = lora_weight

    return merged_params
