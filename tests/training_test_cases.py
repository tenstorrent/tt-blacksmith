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
            "timeout": 2500,
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
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/llama/xla/test_llama_fine_tuning_pure_torch.py",
            "experiment_config": "blacksmith/experiments/torch/llama/xla/lora/single_chip/test_llama_3_2_1b_sst2.yaml",
            "timeout": 1000,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.torch,
            pytest.mark.single_chip,
        ],
        id="llama-single-chip-torch",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/llama/xla/test_llama_fine_tuning_pure_torch.py",
            "experiment_config": "blacksmith/experiments/torch/llama/xla/lora/quietbox/test_llama_3_2_1b.yaml",
            "timeout": 2000,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n300_llmbox,
            pytest.mark.torch,
            pytest.mark.data_parallel,
        ],
        id="llama-data-parallel-quietbox-torch",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/qwen/test_qwen_finetuning.py",
            "experiment_config": "blacksmith/experiments/torch/qwen/test_qwen_1-5b_finetuning.yaml",
            "timeout": 1000,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.torch,
            pytest.mark.single_chip,
        ],
        id="qwen-single-chip-torch",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/gemma11/test_gemma11_finetuning.py",
            "experiment_config": "blacksmith/experiments/torch/gemma11/test_gemma11_finetuning_squadV2.yaml",
            "timeout": 1000,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.torch,
            pytest.mark.single_chip,
        ],
        id="gemma11-single-chip-torch",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/albert/test_albert_finetuning.py",
            "timeout": 1200,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.torch,
            pytest.mark.single_chip,
        ],
        id="albert-single-chip-torch-adapters",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/phi/test_phi_finetuning.py",
            "experiment_config": "blacksmith/experiments/torch/phi/test_phi1_finetuning_sst2.yaml",
            "timeout": 700,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.torch,
            pytest.mark.single_chip,
        ],
        id="phi1-single-chip-torch",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/nerf/test_nerf.py",
            "experiment_config": "blacksmith/experiments/jax/nerf/test_nerf.yaml",
            "timeout": 2000,
        },
        marks=[
            pytest.mark.skip(reason="Jax tests are not supported yet."),
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.jax,
            pytest.mark.single_chip,
        ],
        id="nerf-single-chip-jax",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/llama_dora/test_llama_fine_tuning_jax.py",
            "experiment_config": "blacksmith/experiments/jax/llama_dora/test_llama_fine_tuning_jax.yaml",
            "timeout": 2000,
        },
        marks=[
            pytest.mark.skip(reason="Jax tests are not supported yet."),
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.jax,
            pytest.mark.single_chip,
        ],
        id="llama-dora-single-chip-jax",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/distil_bert/single_chip/test_distil_bert_flax.py",
            "timeout": 2000,
        },
        marks=[
            pytest.mark.skip(reason="Jax tests are not supported yet."),
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.jax,
            pytest.mark.single_chip,
        ],
        id="distilbert-single-chip-jax",
    ),
]
