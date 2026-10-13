# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Portions of this file are derived from 'lorax' by davisyoshida (MIT).
Copyright (c) 2023 davisyoshida
Source: https://github.com/davisyoshida/lorax
See THIRD_PARTY_NOTICES.md for the full MIT license text.
"""
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
import jax.lax as lax
import jax.numpy as jnp
import quax


def dora(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Alias for quax.quaxify to reduce necessary modification to code
    using older version of Lorax
    """
    return quax.quaxify(f)


@dataclass
class DoraWeight(quax.ArrayValue):
    w: jax.Array  # M x N
    a: jax.Array  # k x N
    b: jax.Array  # M x k
    m: jax.Array  # 1 x N
    alpha: float = 1.0

    def __post_init__(self):
        assert self.a.shape[-2] == self.b.shape[-1]
        assert self.w.shape[-2] == self.b.shape[-2]
        assert self.w.shape[-1] == self.a.shape[-1] == self.m.shape[-1]

    def materialise(self):
        v = self.w + self.b @ self.a  # adapted direction
        col_norms = jnp.linalg.norm(v, ord=2, axis=0, keepdims=True)  # (1, N)
        v_normed = v / col_norms  # normalize columns
        return (self.m * v_normed).astype(self.w.dtype)  # scale by m

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.w.shape, self.w.dtype)


@quax.register(lax.dot_general_p)
def handle_dot_lhs_dora(dora: DoraWeight, rhs: jax.Array, *, dimension_numbers: Any, **kwargs: Any) -> Any:
    """
    Handle DoRA forward pass: y = (m * normalize(W + B @ A)) @ x
    """
    if isinstance(rhs, DoraWeight):
        warnings.warn("Encountered product of two DoraWeights. Materializing the rhs")
        rhs = rhs.materialise()

    op = partial(jax.lax.dot_general, **kwargs)

    # First compute adapted direction: W + B @ A
    v = dora.w + (dora.b @ dora.a)
    col_norms = jnp.linalg.norm(v, ord=2, axis=0, keepdims=True)
    v_normed = v / col_norms

    # Scale by magnitude vector (1×N) along columns
    v_scaled = dora.m * v_normed

    # Perform dot product with rhs
    out = op(v_scaled, rhs, dimension_numbers=dimension_numbers)

    return out.astype(dora.w.dtype)


@quax.register(lax.dot_general_p)
def handle_dot_rhs_dora(lhs: jax.Array, dora: DoraWeight, *, dimension_numbers: Any, **kwargs: Any) -> Any:
    """
    Handle DoRA forward pass for x @ (m * normalize(W + B @ A))
    """
    op = partial(jax.lax.dot_general, **kwargs)

    v = dora.w + (dora.b @ dora.a)
    col_norms = jnp.linalg.norm(v, ord=2, axis=0, keepdims=True)
    v_normed = v / col_norms
    v_scaled = dora.m * v_normed

    out = op(lhs, v_scaled, dimension_numbers=dimension_numbers)
    return out.astype(dora.w.dtype)


@quax.register(lax.transpose_p)
def eval_dora_transpose(arg: DoraWeight, *, permutation: Any) -> Any:
    """
    Define how `DoraWeight` behaves under transpose.
    """
    if not (len(arg.shape) == 2 and permutation == (1, 0)):
        return NotImplemented

    return DoraWeight(
        w=arg.w.T,
        a=arg.b.T,
        b=arg.a.T,
        m=arg.m,  # magnitude vector stays the same for transpose
        alpha=arg.alpha,
    )


@quax.register(lax.convert_element_type_p)
def eval_dora_convert_element_type(arg: DoraWeight, *, new_dtype: Any, **_) -> DoraWeight:
    """
    Define dtype conversion for `DoraWeight`.
    """
    return DoraWeight(
        w=jax.lax.convert_element_type(arg.w, new_dtype),
        a=jax.lax.convert_element_type(arg.a, new_dtype),
        b=jax.lax.convert_element_type(arg.b, new_dtype),
        m=jax.lax.convert_element_type(arg.m, new_dtype),
        alpha=arg.alpha,  # leave alpha as a Python float
    )
