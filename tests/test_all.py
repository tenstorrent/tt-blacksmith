# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
from pathlib import Path

import google.protobuf.message
import pandas as pd
import pytest
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore
from training_test_cases import TRAINING_TEST_CASES


def get_history_from_wandb_file(wandb_file: Path) -> pd.DataFrame:
    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))

    train_loss = []
    val_loss = []
    _step = []

    for record_bytes in iter(lambda: ds.scan_record(), None):
        try:
            pb = wandb_internal_pb2.Record()
            pb.ParseFromString(record_bytes[1])
            if pb.HasField("history"):
                for item in pb.history.item:
                    if item.nested_key[0] == "train/loss":
                        train_loss.append(float(item.value_json))
                    elif item.nested_key[0] == "val/loss":
                        val_loss.append(float(item.value_json))
                    elif item.nested_key[0] == "_step":
                        _step.append(int(item.value_json))

        except google.protobuf.message.DecodeError as e:
            print(f"Error decoding record: \n{e}\n")

    return pd.DataFrame({"_step": _step, "train/loss": train_loss, "val/loss": val_loss})


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


    wandb_dir = Path("./wandb/latest-run")
    wandb_files = list(wandb_dir.glob("*.wandb"))
    assert len(wandb_files) > 0, "No .wandb files found in latest-run directory"
    wandb_file = wandb_files[0]

    history = get_history_from_wandb_file(wandb_file)
    golden_file = Path(f"tests/golden_files/{test_id}.csv")

    if golden_file.exists():
        # Golden file already exists, so we can compare the history to it.
        golden_history = pd.read_csv(golden_file, index_col=0)
        pd.testing.assert_frame_equal(history, golden_history, rtol=setup_dict["tolerance"])
    # else:
    # Golden file doesn't exist, so we can save the history to it.
    # TODO: Uncomment this when on GPU.
    # history.to_csv(golden_file)
