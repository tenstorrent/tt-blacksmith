import pytest

# Test cases with individual marks for each configuration
TRAINING_TEST_CASES = [
    pytest.param(
        "blacksmith/experiments/torch/mnist/tensor_parallel/test_mnist_training.py",
        "tests/configs/test_mnist_training_fast.yaml",
        600,
        marks=[
            pytest.mark.skip,
            pytest.mark.uplift,
            pytest.mark.push,
            pytest.mark.n300,
            pytest.mark.torch,
            pytest.mark.tensor_parallel,
        ],
        id="mnist-tensor-parallel"
    ),
    pytest.param(
        "blacksmith/experiments/torch/mnist/data_parallel/test_mnist_training.py",
        "tests/configs/test_mnist_training_fast.yaml",
        600,
        marks=[
            pytest.mark.skip,
            pytest.mark.uplift,
            pytest.mark.push,
            pytest.mark.n300,
            pytest.mark.torch,
            pytest.mark.data_parallel,
        ],
        id="mnist-data-parallel"
    ),
    pytest.param(
        "blacksmith/experiments/torch/mnist/test_mnist_training.py",
        "tests/configs/test_mnist_training_fast.yaml",
        300,
        marks=[
            pytest.mark.uplift,
            pytest.mark.push,
            pytest.mark.n300,
            pytest.mark.torch,
            pytest.mark.single_chip,
        ],
        id="mnist-single-chip"
    ),
]