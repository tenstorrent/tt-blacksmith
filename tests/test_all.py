# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import pytest
import pandas as pd
import wandb
from training_test_cases import TRAINING_TEST_CASES


@pytest.mark.parametrize("setup_dict", TRAINING_TEST_CASES)
def test_training_script(
    setup_dict: dict,
    request: pytest.FixtureRequest,
):
    """
    Test that training script runs successfully with test configuration.

    Spawns subprocess to execute training script, verifies exit code 0.

    Args:
        setup_dict: Dictionary containing the test setup:
            - test_script: Path to the training script.
            - experiment_config: Path to the experiment configuration.
            - test_config: Path to the test configuration.
            - tolerance: Tolerance for loss and accuracy metrics.
            - timeout: Timeout in seconds.
        request: pytest request object.
    """

    default_setup_dict = {
        "test_script": None,
        "experiment_config": None,
        "test_config": "tests/configs/test_training_fast.yaml",
        "tolerance": 0.1,
        "timeout": 800.0,
    }

    setup_dict = default_setup_dict | setup_dict

    test_id = request.node.callspec.id

    assert Path(setup_dict["test_script"]).exists(), f"Script not found: {setup_dict['test_script']}"
    assert Path(setup_dict["test_config"]).exists(), f"Config not found: {setup_dict['test_config']}"

    cmd = [sys.executable, str(setup_dict["test_script"]), "--test-config", str(setup_dict["test_config"])]
    if setup_dict["experiment_config"] is not None:
        cmd.append("--config")
        cmd.append(str(setup_dict["experiment_config"]))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            timeout=setup_dict["timeout"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print(f"\n{'='*60}")
            print(f"FAILED: {test_id}")
            print(f"Exit code: {result.returncode}")
            print(f"\nSTDOUT:\n{result.stdout}")
            print(f"\nSTDERR:\n{result.stderr}")
            print(f"{'='*60}\n")
            pytest.fail(f"Training script exited with code {result.returncode}")

    except subprocess.TimeoutExpired:
        pytest.fail(f"Training script timed out after {setup_dict['timeout']} seconds")

    api = wandb.Api()
    runs = api.runs("test-all-wandb-project")
    run = runs[len(runs) - 1]
    # We use "_step" here in order to support parity for trainings with checkpoints
    # where _step doesn't match the row number.
    history = run.history()[["_step", "train/loss", "val/loss"]]
    golden_file = Path(f"tests/golden_files/{run.name}_history.csv")

    if golden_file.exists():
        # Golden file already exists, so we can compare the history to it.
        golden_history = pd.read_csv(golden_file, index_col=0)
        pd.testing.assert_frame_equal(history, golden_history, rtol=setup_dict["tolerance"])
    # else:
        # Golden file doesn't exist, so we can save the history to it.
        # TODO: Uncomment this when on GPU.
        # history.to_csv(golden_file)
