# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

# Test cases with individual marks for each configuration
TRAINING_TEST_CASES = [
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/mnist/tensor_parallel/test_mnist_training.py",
            "experiment_config": "blacksmith/experiments/torch/mnist/tensor_parallel/test_mnist_training_tp.yaml",
            "timeout": 300,
        },
        marks=[
            pytest.mark.push,
            pytest.mark.n300,
            pytest.mark.torch,
            pytest.mark.tensor_parallel,
        ],
        id="tt-mlp-mnist-n300-tp",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/mnist/data_parallel/test_mnist_training.py",
            "experiment_config": "blacksmith/experiments/torch/mnist/data_parallel/test_mnist_training_dp.yaml",
            "timeout": 300,
        },
        marks=[
            pytest.mark.push,
            pytest.mark.n300,
            pytest.mark.torch,
            pytest.mark.data_parallel,
        ],
        id="tt-mlp-mnist-n300-dp",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/mnist/test_mnist_training.py",
            "experiment_config": "blacksmith/experiments/torch/mnist/test_mnist_training.yaml",
            "timeout": 300,
        },
        marks=[
            pytest.mark.push,
            pytest.mark.n150,
            pytest.mark.n300,
            pytest.mark.torch,
            pytest.mark.single_chip,
        ],
        id="tt-mlp-mnist-n150",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/mnist/multi_chip/data_parallel/test_pure_jax_mnist.py",
            "experiment_config": "blacksmith/experiments/jax/mnist/test_mnist.yaml",
            "timeout": 3000,
            "skip_loss_checks": True,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n300,
            pytest.mark.jax,
            pytest.mark.data_parallel,
        ],
        id="tt-mlp-mnist-n300-dp-jax",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/mnist/single_chip/test_pure_jax_mnist.py",
            "experiment_config": "blacksmith/experiments/jax/mnist/test_mnist.yaml",
            "timeout": 200,
            "skip_loss_checks": True,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.jax,
            pytest.mark.single_chip,
        ],
        id="tt-mlp-mnist-n150-jax",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/mnist/single_chip/test_flax_mnist.py",
            "experiment_config": "blacksmith/experiments/jax/mnist/test_mnist.yaml",
            "timeout": 400,
            "skip_loss_checks": True,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.jax,
            pytest.mark.single_chip,
        ],
        id="tt-mlp-mnist-flax-n150-jax",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/mnist/multi_chip/tensor_parallel/test_pure_jax_mnist.py",
            "experiment_config": "blacksmith/experiments/jax/mnist/test_mnist.yaml",
            "timeout": 3000,
            "skip_loss_checks": True,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n300,
            pytest.mark.jax,
            pytest.mark.tensor_parallel,
        ],
        id="tt-mlp-mnist-n300-tp-jax",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/llama/xla/test_llama_fine_tuning_pure_torch.py",
            "experiment_config": "blacksmith/experiments/torch/llama/xla/lora/single_chip/test_llama_3_2_1b_sst2.yaml",
            "timeout": 3000,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.torch,
            pytest.mark.single_chip,
            pytest.mark.split_0,
        ],
        id="tt-llama_3_2_1b-sst2-n150",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/llama/xla/test_llama_fine_tuning_pure_torch.py",
            "experiment_config": "blacksmith/experiments/torch/llama/xla/lora/quietbox/test_llama_3_2_1b.yaml",
            "timeout": 3000,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n300_llmbox,
            pytest.mark.torch,
            pytest.mark.data_parallel,
        ],
        id="tt-llama_3_2_1b-sst2-n300-llmbox",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/qwen/test_qwen_finetuning.py",
            "experiment_config": "blacksmith/experiments/torch/qwen/single_chip/test_qwen_1-5b_finetuning.yaml",
            "timeout": 2000,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.torch,
            pytest.mark.single_chip,
            pytest.mark.split_0,
        ],
        id="tt-qwen_1_5b-text2sql-n150",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/gemma11/test_gemma11_finetuning.py",
            "experiment_config": "blacksmith/experiments/torch/gemma11/test_gemma11_finetuning_squadV2.yaml",
            "timeout": 10000,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.torch,
            pytest.mark.single_chip,
            pytest.mark.split_1,
        ],
        id="tt-gemma11-squadv2-n150",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/albert/test_albert_finetuning.py",
            "experiment_config": "blacksmith/experiments/torch/albert/test_albert_finetuning.yaml",
            "test_config": "tests/configs/tt-albert_base_v2-banking77-n150.yaml",
            "timeout": 3600,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.torch,
            pytest.mark.single_chip,
            pytest.mark.split_1,
        ],
        id="tt-albert_base_v2-banking77-n150",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/torch/phi/test_phi_finetuning.py",
            "experiment_config": "blacksmith/experiments/torch/phi/test_phi1_finetuning_sst2.yaml",
            "timeout": 7000,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.torch,
            pytest.mark.single_chip,
        ],
        id="tt-phi1-sst2-n150",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/nerf/test_nerf.py",
            "experiment_config": "blacksmith/experiments/jax/nerf/test_nerf.yaml",
            "timeout": 20000,
            "skip_loss_checks": True,
        },
        marks=[
            pytest.mark.skip(reason="Jax tests are not supported yet."),
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.jax,
            pytest.mark.single_chip,
        ],
        id="tt-nerf-nerf-p150-white-n150-jax",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/llama_dora/test_llama_fine_tuning_jax.py",
            "experiment_config": "blacksmith/experiments/jax/llama_dora/test_llama_fine_tuning_jax.yaml",
            "timeout": 20000,
            "skip_loss_checks": True,
        },
        marks=[
            pytest.mark.skip(reason="Jax tests are not supported yet."),
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.jax,
            pytest.mark.single_chip,
        ],
        id="tt-llama_3_2_1b-dora-glue-sst2-n150-jax",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/experiments/jax/distil_bert/single_chip/test_distil_bert_flax.py",
            "experiment_config": "blacksmith/experiments/jax/distil_bert/test_distil_bert_flax.yaml",
            "timeout": 20000,
            "skip_loss_checks": True,
        },
        marks=[
            pytest.mark.skip(reason="Jax tests are not supported yet."),
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.jax,
            pytest.mark.single_chip,
        ],
        id="tt-distilbert-glue-sst2-n150-jax",
    ),
]
