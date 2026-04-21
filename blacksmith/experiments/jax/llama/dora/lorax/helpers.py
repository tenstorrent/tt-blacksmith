# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Portions of this file are derived from 'lorax' by davisyoshida (MIT).
Copyright (c) 2023 davisyoshida
Source: https://github.com/davisyoshida/lorax
See THIRD_PARTY_NOTICES.md for the full MIT license text.
"""
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import DictKey, SequenceKey, tree_map_with_path

from .constants import DORA_FREEZE, DORA_FULL
from .transform import DoraWeight


def init_dora(param_tree, spec, rng, stddev=0.01, dtype=jnp.float32, alpha=1.0, is_leaf=None) -> Any:
    def iter_keys(key):
        # Infinite PRNGKey generator: repeatedly split the key to obtain a fresh
        # subkey for initializing each DoRA adapter tensor.
        while True:
            key, out_key = jax.random.split(key)
            yield out_key

    key_it = iter_keys(rng)

    def get_param(path, param, spec_val):
        # Map a single parameter and spec value to either:
        # - the original parameter (freeze/full tune), or
        # - a DoraWeight wrapping the base weight with newly initialized A/B.
        if spec_val in (DORA_FREEZE, DORA_FULL):
            return param

        if len(param.shape) == 1:
            raise ValueError(
                f"Vectors must either be frozen or fully tuned, but got spec value {spec} for param with path {path}"
            )

        if len(param.shape) == 2:
            b_dim, a_dim = param.shape

            b = jnp.zeros((b_dim, spec_val), dtype=dtype)
            a = jax.random.normal(next(key_it), (spec_val, a_dim), dtype=dtype) * stddev

            # Initialize magnitude vector for DoRA - use column norms of original weight
            col_norms = jnp.linalg.norm(param, ord=2, axis=0, keepdims=True)
            m = col_norms.astype(dtype)

            return DoraWeight(w=param, a=a, b=b, m=m, alpha=alpha)

        # conv case
        *window_shape, in_channels, out_channels = param.shape

        a = jnp.zeros(
            (*(1 for _ in range(len(window_shape))), spec_val, out_channels),
            dtype=param.dtype,
        )
        b = jax.random.normal(rng, (*window_shape, in_channels, spec_val), dtype=param.dtype) * stddev

        # Initialize magnitude vector for DoRA - use column norms of original weight (flattened)
        axes_to_norm = tuple(range(len(param.shape) - 1))  # All axes except output channels
        col_norms = jnp.linalg.norm(param, ord=2, axis=axes_to_norm, keepdims=True)
        m_shape = (*(1 for _ in range(len(window_shape))), 1, out_channels)
        m = col_norms.reshape(m_shape).astype(param.dtype)

        return DoraWeight(param, a, b, m, alpha=alpha)

    return jax.tree_util.tree_map_with_path(get_param, param_tree, spec, is_leaf=is_leaf)


def simple_spec(params, decision_fn=None, tune_vectors=False, is_leaf=None) -> Any:
    """
    Create a simple DoRA spec for a pytree.
    Args:
        params: pytree of parameters
        tune_vectors: If true, will flag all arrays with less than 2 dimensions for tuning
        decision_fn: A function which maps a Jax KeyPath and a parameter to a spec value
    """
    if decision_fn is None:

        def decision_fn(*args):
            return DORA_FREEZE

    def full_fn(path, arr):
        if len(arr.shape) < 2:
            return DORA_FULL if tune_vectors else DORA_FREEZE

        value = decision_fn(path, arr)
        return value

    return tree_map_with_path(full_fn, params, is_leaf=is_leaf)


def split_trainable_frozen(dora_params, dora_spec) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split DoRA parameters into trainable and frozen pytrees.
    Trainable: Only DoRA a,b matrices from DoraWeight objects
    Frozen: Everything else (original weights w, alpha, and regular frozen params)
    """
    trainable_params = {}
    frozen_params = {}

    def split_param(param, spec_value, path_parts):
        if isinstance(param, DoraWeight):
            # For DoraWeight: only a,b are trainable, m should also be trainable for DoRA
            trainable_params[".".join(path_parts)] = {"a": param.a, "b": param.b, "m": param.m}
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

    traverse_tree(dora_params, dora_spec)

    print(f"Split completed:")
    print(f" Trainable params: {len(trainable_params)} DoRA matrix pairs (a, b, m)")
    print(f" Frozen params: {len(frozen_params)} weight groups")

    return trainable_params, frozen_params


def merge_trainable_frozen(trainable_params, frozen_params) -> Any:
    """
    Merge trainable and frozen pytrees back into full DoRA parameter tree.
    """
    merged_params = {}

    # First add all frozen regular parameters
    for path, param in frozen_params.items():
        if isinstance(param, dict) and "w" in param and "alpha" in param:
            # This is a frozen DoraWeight component - will be merged with trainable
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

    # Now merge DoraWeight objects
    for path, trainable in trainable_params.items():
        frozen_dora = frozen_params[path]  # Should have 'w' and 'alpha'

        # Reconstruct DoraWeight
        dora_weight = DoraWeight(
            w=frozen_dora["w"], a=trainable["a"], b=trainable["b"], m=trainable["m"], alpha=frozen_dora["alpha"]
        )

        # Place in merged tree
        keys = path.split(".")
        current = merged_params
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = dora_weight

    return merged_params
