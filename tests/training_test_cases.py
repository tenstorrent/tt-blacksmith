# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

# Test cases with individual marks for each configuration. A crank entry looks like this:
#
# pytest.param(
#     {
#         "test_script": "blacksmith/tools/trainer/examples/crank/llm/train.py",
#         "experiment_config": "blacksmith/tools/trainer/examples/crank/llm/lora/single_chip/llama_3_2_1b_sst2.yaml",
#         "test_config": "tests/configs/tt-llama_3_2_1b-sst2-n150-trainer.yaml",
#         "timeout": 5000,
#     },
#     marks=[pytest.mark.uplift, pytest.mark.n150, pytest.mark.torch, pytest.mark.single_chip],
#     id="tt-llama_3_2_1b-sst2-n150-trainer",
# ),
TRAINING_TEST_CASES = [
    # ttml (tt-train) LoRA on SST-2 (1B: 32 steps, 8B: 500 steps with checkpoints); run in the ttml env (`source env/activate --ttml`).
    pytest.param(
        {
            "test_script": "blacksmith/tools/trainer/examples/ttml/llm/train.py",
            "experiment_config": "blacksmith/tools/trainer/examples/ttml/llm/lora/single_chip/llama_3_2_1b_sst2.yaml",
            "test_config": "tests/configs/tt-llama_3_2_1b-sst2-p150-ttml.yaml",
            "timeout": 1200,
        },
        marks=[pytest.mark.push, pytest.mark.p150, pytest.mark.ttml, pytest.mark.single_chip],
        id="tt-llama_3_2_1b-sst2-p150-ttml",
    ),
    pytest.param(
        {
            "test_script": "blacksmith/tools/trainer/examples/ttml/llm/train.py",
            "experiment_config": "blacksmith/tools/trainer/examples/ttml/llm/lora/single_chip/llama_3_1_8b_sst2.yaml",
            "test_config": "tests/configs/tt-llama_3_1_8b-sst2-p150-ttml.yaml",
            "timeout": 3600,
        },
        marks=[pytest.mark.uplift, pytest.mark.p150, pytest.mark.ttml, pytest.mark.single_chip],
        id="tt-llama_3_1_8b-sst2-p150-ttml",
    ),
    # Starts from the step-250 PEFT adapter of the case above: checks that loading the adapter
    # restores the trained model (the first validation loss equals that run's at step 250).
    pytest.param(
        {
            "test_script": "blacksmith/tools/trainer/examples/ttml/llm/train.py",
            "experiment_config": "blacksmith/tools/trainer/examples/ttml/llm/lora/single_chip/llama_3_1_8b_sst2.yaml",
            "test_config": "tests/configs/tt-llama_3_1_8b-sst2-p150-ttml-init_step250.yaml",
            "test_checkpoint_path": "tests/checkpoints/tt-llama_3_1_8b-sst2-p150-ttml_checkpoint_step250",
            "timeout": 3600,
        },
        marks=[pytest.mark.uplift, pytest.mark.p150, pytest.mark.ttml, pytest.mark.single_chip],
        id="tt-llama_3_1_8b-sst2-p150-ttml-init_step250",
    ),
]
