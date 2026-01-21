# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for training scripts."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import wandb
from training_test_cases import TRAINING_TEST_CASES


def assert_wandb_history_equals_with_tolerance(
    history,
    golden_history,
    tolerances
):
    """
    Assert that the history is equal to the golden history with a tolerance.
    """
    for col, tolerance in tolerances.items():
        diff = (history[col] - golden_history[col]).abs()
        ok = (diff <= tolerance).all()

        if not ok:
            # TODO: (agobeljicTT) Add a more detailed explanation of the failure.
            pytest.fail(
                f"Wandb history is not equal to the golden history with a tolerance of {tolerance} for column {col}"
            )


@pytest.mark.parametrize("test_script,test_config,timeout,tolerances", TRAINING_TEST_CASES)
def test_training_script(test_script, test_config, timeout, tolerances, request):
    """
    Test that training script runs successfully with test configuration.

    Spawns subprocess to execute training script, verifies exit code 0.
    """

    test_id = request.node.callspec.id
    golden_csv_path = Path(f"tests/golden_files/{test_id}.csv")

    # Verify files exist
    assert Path(test_script).exists(), f"Script not found: {test_script}"
    assert Path(test_config).exists(), f"Config not found: {test_config}"

    # Build command
    cmd = [sys.executable, str(test_script), "--test-config", str(test_config)]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )

        # Check exit code
        if result.returncode != 0:
            print(f"\n{'='*60}")
            print(f"FAILED: {test_id}")
            print(f"Exit code: {result.returncode}")
            print(f"\nSTDOUT:\n{result.stdout}")
            print(f"\nSTDERR:\n{result.stderr}")
            print(f"{'='*60}\n")
            pytest.fail(f"Training script exited with code {result.returncode}")

        # Get the last run from the project
        # api = wandb.Api()
        # runs = api.runs("test-all-wandb-project") # Universally unique identifier
        # run = runs[len(runs) - 1]
        # history = run.history()

        # TODO: (agobeljicTT) Add golden file functionality.
        # If the golden CSV file exists, compare the history to the golden CSV file
        # if golden_csv_path.exists():
        #    golden_history = pd.read_csv(golden_csv_path)
        #    tolerances = {
        #"_runtime": 0.1,
        #"val/loss": 0.002,
        #"val/accuracy": 0.002,
        #"train/loss": 0.002,
        #    } | tolerances
        #    assert_wandb_history_equals_with_tolerance(history, golden_history, tolerances)
        #
        # If the golden CSV file does not exist, save the history to the golden CSV file
        # else:
        #    history.to_csv(golden_csv_path, index=False)
        #    pytest.skip(f"Golden CSV file not found, generated new one: {golden_csv_path}. Please rerun the test.")

    except subprocess.TimeoutExpired:
        pytest.fail(f"Training script timed out after {timeout} seconds")
