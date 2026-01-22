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


@pytest.mark.parametrize("test_script,test_config,timeout", TRAINING_TEST_CASES)
def test_training_script(test_script, test_config, timeout, request):
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
        api = wandb.Api()
        runs = api.runs("test-all-wandb-project")  # Universally unique identifier
        run = runs[len(runs) - 1]
        history = run.history()  # TODO: (agobeljicTT) For Superset

    except subprocess.TimeoutExpired:
        pytest.fail(f"Training script timed out after {timeout} seconds")
