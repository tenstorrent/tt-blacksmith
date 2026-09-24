# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

# Test cases with individual marks for each configuration.

TRAINING_TEST_CASES = [
    pytest.param(
        {
            "test_script": "blacksmith/experiments/llama/train.py",
            "experiment_config": "blacksmith/experiments/llama/lora/single_chip/llama_3_2_1b_sst2.yaml",
            "test_config": "tests/configs/tt-llama_3_2_1b-sst2-n150.yaml",
            "timeout": 5000,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.n150,
            pytest.mark.torch,
            pytest.mark.single_chip,
        ],
        id="tt-llama_3_2_1b-sst2-n150",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/tools/trainer/examples/lora_llm/train.py",
            "experiment_config": "blacksmith/tools/trainer/examples/lora_llm/single_chip/llama_3_1_8b_sst2.yaml",
            "timeout": 5000,
        },
        marks=[
            pytest.mark.uplift,
            pytest.mark.p150,
            pytest.mark.torch,
            pytest.mark.single_chip,
        ],
        id="tt-llama_3_1_8b-sst2-p150-trainer",
    ),
]
