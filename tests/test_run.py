"""Integration tests for training scripts."""
import pytest
import subprocess
import sys
from pathlib import Path


# Test cases: (config_name, script_path, timeout_seconds, test_id)
TRAINING_TEST_CASES = [
    (
        "test_mnist_training_tp_fast.yaml",
        "blacksmith/experiments/torch/mnist/tensor_parallel/test_mnist_training.py",
        600,
        "mnist-tensor-parallel"
    ),
    (
        "test_mnist_training_dp_fast.yaml",
        "blacksmith/experiments/torch/mnist/data_parallel/test_mnist_training.py",
        600,
        "mnist-data-parallel"
    ),
    (
        "test_mnist_training_fast.yaml",
        "blacksmith/experiments/torch/mnist/test_mnist_training.py",
        300,
        "mnist-single-chip"
    ),
]


@pytest.mark.parametrize(
    "config_name,script_path,timeout,test_id",
    TRAINING_TEST_CASES,
    ids=[case[3] for case in TRAINING_TEST_CASES]
)
def test_training_script(repo_root, test_configs_dir, config_name, script_path, timeout, test_id):
    """
    Test that training script runs successfully with test configuration.

    Spawns subprocess to execute training script, verifies exit code 0.
    """
    config_path = test_configs_dir / config_name
    script_full_path = repo_root / script_path

    # Verify files exist
    assert config_path.exists(), f"Config not found: {config_path}"
    assert script_full_path.exists(), f"Script not found: {script_full_path}"

    # Build command
    cmd = [
        sys.executable,
        str(script_full_path),
        "--config",
        str(config_path)
    ]
    print(cmd)

    # Execute subprocess
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False
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

        # Verify success message in output
        assert "Training finished successfully" in result.stdout or \
               "finished successfully" in result.stdout.lower(), \
               "Expected success message not found in output"

    except subprocess.TimeoutExpired:
        pytest.fail(f"Training script timed out after {timeout} seconds")
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")
