# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pickle
import glob
import os

# Checkpointing utilities.
def save_checkpoint(checkpoint_dir, step, trainable_params, opt_state, rng):
    # Save checkpoint with student params (only trainable), optimizer state, and training metadata.
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pkl")

    checkpoint = {"step": step, "trainable_params": trainable_params, "opt_state": opt_state, "rng": rng}

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"Saved checkpoint at step {step} to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path):
    # Load checkpoint and return training state.
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    print(f"Loaded checkpoint from step {checkpoint['step']}")
    return checkpoint


def get_latest_checkpoint(checkpoint_dir):
    # Find the latest checkpoint in directory.
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pkl"))
    if not checkpoints:
        return None

    # Sort by step number.
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return checkpoints[-1]


def cleanup_old_checkpoints(checkpoint_dir, keep_top_k):
    # Remove old checkpoints, keeping only the most recent k.
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pkl"))
    if len(checkpoints) <= keep_top_k:
        return

    # Sort by step number.
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Remove oldest checkpoints.
    for checkpoint in checkpoints[:-keep_top_k]:
        os.remove(checkpoint)
        print(f"Removed old checkpoint: {checkpoint}")
