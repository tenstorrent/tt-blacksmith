"""Integration tests for training scripts."""
import pytest
import wandb
import subprocess
from pathlib import Path
import sys
import os
from training_test_cases import TRAINING_TEST_CASES

@pytest.mark.parametrize(
    "test_script,test_config,timeout",
    TRAINING_TEST_CASES
)
def test_training_script(test_script, test_config, timeout, request):
    """
    Test that training script runs successfully with test configuration.

    Spawns subprocess to execute training script, verifies exit code 0.
    """

    test_id = request.node.callspec.id

    # Verify files exist
    assert Path(test_script).exists(), f"Script not found: {test_script}"
    assert Path(test_config).exists(), f"Config not found: {test_config}"

    # Build command
    cmd = [
        sys.executable,
        str(test_script),
        "--test-config",
        str(test_config)
    ]

    try:
        env = os.environ.copy()
        env["WANDB_MODE"] = "dryrun"

        result = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
            env=env
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

        api = wandb.Api()
        run = api.runs("test-all-wandb-project")[0]
        history = run.history()
        print(f"RANKO: {history}")

    except subprocess.TimeoutExpired:
        pytest.fail(f"Training script timed out after {timeout} seconds")
