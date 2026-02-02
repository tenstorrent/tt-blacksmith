# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

# Test cases with individual marks for each configuration
TRAINING_TEST_CASES = [
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/mnist/tensor_parallel/test_mnist_training.py",
        },
        marks=[
            pytest.mark.push,
            pytest.mark.n300,
            pytest.mark.torch,
            pytest.mark.tensor_parallel,
        ],
        id="mnist-tensor-parallel-torch",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/mnist/data_parallel/test_mnist_training.py",
        },
        marks=[
            pytest.mark.push,
            pytest.mark.n300,
            pytest.mark.torch,
            pytest.mark.data_parallel,
        ],
        id="mnist-data-parallel-torch",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/mnist/test_mnist_training.py",
            "timeout": 300,
        },
        marks=[
            pytest.mark.push,
            pytest.mark.n150,
            pytest.mark.n300,
            pytest.mark.torch,
            pytest.mark.single_chip,
        ],
        id="mnist-single-chip-torch",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/mnist/multi_chip/data_parallel/test_pure_jax_mnist.py",
            "timeout": 2000,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n300,
            pytest.mark.jax,
            pytest.mark.data_parallel,
        ],
        id="mnist-data-parallel-jax",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/mnist/single_chip/test_pure_jax_mnist.py",
            "timeout": 300,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.jax,
            pytest.mark.single_chip,
        ],
        id="mnist-single-chip-jax",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/mnist/single_chip/test_flax_mnist.py",
            "timeout": 300,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.jax,
            pytest.mark.single_chip,
        ],
        id="mnist-single-chip-jax-flax",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/mnist/multi_chip/tensor_parallel/test_pure_jax_mnist.py",
            "timeout": 1400,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n300,
            pytest.mark.jax,
            pytest.mark.tensor_parallel,
        ],
        id="mnist-tensor-parallel-jax",
    ),
]
