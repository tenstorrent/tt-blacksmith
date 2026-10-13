# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest  # noqa: F401

# Test cases with individual marks for each configuration. Empty until the
# tt-crank recipes land; one entry looks like this:
#
# pytest.param(
#     {
#         "test_script": "blacksmith/tools/trainer/examples/lora_llm/train.py",
#         "experiment_config": "blacksmith/tools/trainer/examples/lora_llm/single_chip/llama_3_2_1b_sst2.yaml",
#         "test_config": "tests/configs/tt-llama_3_2_1b-sst2-n150-trainer.yaml",
#         "timeout": 5000,
#     },
#     marks=[pytest.mark.uplift, pytest.mark.n150, pytest.mark.torch, pytest.mark.single_chip],
#     id="tt-llama_3_2_1b-sst2-n150-trainer",
# ),
TRAINING_TEST_CASES = []
