# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import pytest
from training_test_cases import TRAINING_TEST_CASES, Seconds
from typing import Optional


@pytest.mark.parametrize(
    "test_script,experiment_config,test_config,timeout", TRAINING_TEST_CASES
)
def test_training_script(
    test_script: str,
    experiment_config: Optional[str],
    test_config: Optional[str],
    timeout: Seconds,
    request: pytest.FixtureRequest,
):
    """
    Test that training script runs successfully with test configuration.

    Spawns subprocess to execute training script, verifies exit code 0.
    """

    test_id = request.node.callspec.id

    assert Path(test_script).exists(), f"Script not found: {test_script}"
    assert Path(test_config).exists(), f"Config not found: {test_config}"

    if experiment_config is not None:
        cmd = [
            sys.executable,
            str(test_script),
            "--config",
            str(experiment_config),
            "--test-config",
            str(test_config),
        ]
    else:
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

        if result.returncode != 0:
            print(f"\n{'='*60}")
            print(f"FAILED: {test_id}")
            print(f"Exit code: {result.returncode}")
            print(f"\nSTDOUT:\n{result.stdout}")
            print(f"\nSTDERR:\n{result.stderr}")
            print(f"{'='*60}\n")
            pytest.fail(f"Training script exited with code {result.returncode}")

    except subprocess.TimeoutExpired:
        pytest.fail(f"Training script timed out after {timeout} seconds")
