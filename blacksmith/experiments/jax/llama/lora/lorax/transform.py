# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import warnings
from dataclasses import dataclass
from functools import partial

"""
Portions of this file are derived from 'lorax' by davisyoshida (MIT).
Copyright (c) 2023 davisyoshida
Source: https://github.com/davisyoshida/lorax
See THIRD_PARTY_NOTICES.md for the full MIT license text.
"""
from typing import Any, Callable

import jax
import jax.lax as lax
import quax


def lora(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Alias for quax.quaxify to reduce necessary modification to code
    using older version of Lorax
    """
    return quax.quaxify(f)


@dataclass
class LoraWeight(quax.ArrayValue):
    w: jax.Array  # M x N
    a: jax.Array  # k x N
    b: jax.Array  # M x k
    alpha: float = 1.0

    def __post_init__(self):
        assert self.a.shape[-2] == self.b.shape[-1]
        assert self.w.shape[-2] == self.b.shape[-2]
        assert self.w.shape[-1] == self.a.shape[-1]

    def materialise(self):
        return (self.w + self.get_scale() * self.b @ self.a).astype(self.w.dtype)

    def get_scale(self) -> float:
        return self.alpha / self.b.shape[-1]

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.w.shape, self.w.dtype)


def _check_dot_dimension_numbers(dimension_numbers: Any) -> bool:
    # Validate that the `dot_general` call is compatible with LoRA's unbatched
    # matrix multiply semantics. We only support a single contracting dimension
    # and no batch dimensions. Returning True means "supported"; returning
    # False asks Quax to fall back to the default implementation.
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    if lhs_batch or rhs_batch:
        warnings.warn("Lorax does not support batched matmuls")
        return False
    if len(lhs_contract) != 1 or len(rhs_contract) != 1:
        warnings.warn("Lorax only supports matmul")
        return False
    return True


@quax.register(lax.dot_general_p)
def handle_dot_lhs(lora: LoraWeight, rhs: jax.Array, *, dimension_numbers: Any, **kwargs: Any) -> Any:
    # Quax primitive registration: this decorator tells Quax to intercept the
    # JAX primitive `lax.dot_general_p` when the LHS operand is a `LoraWeight`.
    # During tracing/compilation, Quax dispatches to this handler so we can
    # inject the LoRA low‑rank update without materializing (W + B @ A).
    # Mathematically, for (W + (alpha/k) * B @ A) @ X, we compute:
    #   (W @ X) + B @ (A @ X)
    # using two small matmuls, preserving dtype and shapes expected by JAX.
    if not _check_dot_dimension_numbers(dimension_numbers):
        return NotImplemented

    if isinstance(rhs, LoraWeight):
        rhs = rhs.materialise()
        warnings.warn("Encountered product of two LoraWeights. Materializing the rhs")

    op = partial(jax.lax.dot_general, **kwargs)

    (lhs_contract,) = dimension_numbers[0][0]

    first, second = (lora.a, lora.b) if lhs_contract == 1 else (lora.b, lora.a)

    first *= lora.get_scale()

    orig = op(lora.w, rhs, dimension_numbers=dimension_numbers)
    lora_product = op(first, rhs, dimension_numbers=dimension_numbers)

    second_dimension_numbers = ((lhs_contract,), (0,)), dimension_numbers[1]

    lora_product = op(second, lora_product, dimension_numbers=second_dimension_numbers)

    return (orig + lora_product).astype(orig.dtype)


@quax.register(lax.dot_general_p)
def handle_dot_rhs(lhs: jax.Array, lora: LoraWeight, *, dimension_numbers: Any, **kwargs: Any) -> Any:
    # Symmetric to the LHS case: register an implementation for `lax.dot_general_p`
    # when the RHS operand is a `LoraWeight`. For X @ (W + (alpha/k) * B @ A), we
    # compute:
    #   (X @ W) + (X @ B) @ A
    # avoiding materializing the adapted weight while matching JAX semantics.
    if not _check_dot_dimension_numbers(dimension_numbers):
        return NotImplemented
    op = partial(jax.lax.dot_general, **kwargs)

    (rhs_contract,) = dimension_numbers[0][1]
    first, second = (lora.a, lora.b) if rhs_contract == 1 else (lora.b, lora.a)

    first *= lora.get_scale()

    orig = op(lhs, lora.w, dimension_numbers=dimension_numbers)
    lora_product = op(lhs, first, dimension_numbers=dimension_numbers)

    second_dimension_numbers = ((lhs.ndim - 1), (rhs_contract,)), dimension_numbers[1]

    lora_product = op(lora_product, second, dimension_numbers=second_dimension_numbers)

    return (orig + lora_product).astype(orig.dtype)


@quax.register(lax.transpose_p)
def eval_lora_transpose(arg: LoraWeight, *, permutation: Any) -> Any:
    # Define how a `LoraWeight` behaves under transpose. For 2D weights and a
    # simple (1, 0) permutation, return a new `LoraWeight` with all components
    # transposed, preserving LoRA structure without materialization.
    if not len(arg.shape) == 2 and permutation == (1, 0):
        return NotImplemented

    return LoraWeight(
        w=arg.w.T,
        a=arg.b.T,
        b=arg.a.T,
        alpha=arg.alpha,
    )


@quax.register(lax.convert_element_type_p)
def eval_lora_convert_element_type(arg: LoraWeight, *, new_dtype: Any, **_) -> LoraWeight:
    # Define dtype conversion for `LoraWeight`. Convert internal arrays to the
    # requested dtype while keeping `alpha` as a Python float.
    return LoraWeight(
        w=jax.lax.convert_element_type(arg.w, new_dtype),
        a=jax.lax.convert_element_type(arg.a, new_dtype),
        b=jax.lax.convert_element_type(arg.b, new_dtype),
        alpha=arg.alpha,  # leave alpha as a Python float
    )
