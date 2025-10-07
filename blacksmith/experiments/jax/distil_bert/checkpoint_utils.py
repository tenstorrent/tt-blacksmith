# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pickle
from pathlib import Path

# Checkpointing utilities.
def save_checkpoint(checkpoint_dir, step, trainable_params, opt_state, rng):
    # Save checkpoint with student params (only trainable), optimizer state, and training metadata.
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_path / f"checkpoint_{step}.pkl"

    checkpoint = {"step": step, "trainable_params": trainable_params, "opt_state": opt_state, "rng": rng}

    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"Saved checkpoint at step {step} to {checkpoint_file}")
    return checkpoint_file


def load_checkpoint(checkpoint_path):
    # Load checkpoint and return training state.
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    print(f"Loaded checkpoint from step {checkpoint['step']}")
    return checkpoint


def get_latest_checkpoint(checkpoint_dir):
    # Find the path of the latest checkpoint in the directory.
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None

    checkpoints = list(checkpoint_path.glob("checkpoint_*.pkl"))
    if not checkpoints:
        return None

    # Sort by step number.
    checkpoints.sort(key=lambda x: int(x.name.split("_")[-1].split(".")[0]))
    return checkpoints[-1]


def cleanup_old_checkpoints(checkpoint_dir, keep_top_k):
    # Remove old checkpoints, keeping only the most recent k.
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = list(checkpoint_path.glob("checkpoint_*.pkl"))
    if len(checkpoints) <= keep_top_k:
        return

    # Sort by step number.
    checkpoints.sort(key=lambda x: int(x.name.split("_")[-1].split(".")[0]))

    # Remove oldest checkpoints.
    checkpoints_to_remove = checkpoints if keep_top_k == 0 else checkpoints[:-keep_top_k]
    for checkpoint in checkpoints_to_remove:
        checkpoint.unlink()
        print(f"Removed old checkpoint: {checkpoint}")
