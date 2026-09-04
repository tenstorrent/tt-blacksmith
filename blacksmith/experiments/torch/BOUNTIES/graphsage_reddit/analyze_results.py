# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Reproduce GraphSAGE CPU/TT summary metrics and training curves."""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TIMESTAMP = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[,.]\d+)?)"
# tqdm progress updates and logger output can share a physical line in captured
# terminals, so the timestamp is not always at the beginning of a line.
LOG_PREFIX = r"\d{4}-\d{2}-\d{2}[^\n]*?\| INFO \| "
STEP_RE = re.compile(
    rf"{LOG_PREFIX}Step (\d+) \| train/loss: ({FLOAT}) \| "
    rf"train/model_step_time_s: ({FLOAT}) \| train/seed_nodes_per_s: ({FLOAT})",
    re.MULTILINE,
)
FIRST_STEP_RE = re.compile(
    rf"{LOG_PREFIX}Step 1 \| train/compile_and_first_step_time_s: ({FLOAT})",
    re.MULTILINE,
)
INITIAL_RE = re.compile(
    rf"{LOG_PREFIX}Step 0 \| val/loss: ({FLOAT}) \| val/acc: ({FLOAT})",
    re.MULTILINE,
)
EPOCH_RE = re.compile(
    rf"{LOG_PREFIX}Step (\d+) \| train/epoch_loss: ({FLOAT}) \| " rf"val/loss: ({FLOAT}) \| val/acc: ({FLOAT})",
    re.MULTILINE,
)
TEST_LOSS_RE = re.compile(rf"{LOG_PREFIX}\s+test/loss: ({FLOAT})", re.MULTILINE)
TEST_ACC_RE = re.compile(rf"{LOG_PREFIX}\s+test/acc: ({FLOAT})", re.MULTILINE)
TRAIN_START_TIME_RE = re.compile(rf"{TIMESTAMP} \| INFO \| Starting training\.\.\.")
TEST_ACC_TIME_RE = re.compile(rf"{TIMESTAMP} \| INFO \|\s+test/acc: {FLOAT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, required=True, help="CPU run artifact directory")
    parser.add_argument("--tt-dir", type=Path, required=True, help="TT run artifact directory")
    parser.add_argument("--output-dir", type=Path, default=Path("graphsage-analysis"))
    parser.add_argument("--cpu-label", default="CPU (SpMM)")
    parser.add_argument("--tt-label", default="Wormhole TT")
    return parser.parse_args()


def find_one(directory: Path, pattern: str, description: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {description} matching {pattern!r} in {directory}, found {len(matches)}")
    return matches[0]


def require_match(pattern: re.Pattern[str], text: str, description: str) -> re.Match[str]:
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Could not find {description} in the raw log")
    return match


def read_metric_csv(path: Path, value_column: str) -> tuple[list[int], list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "_step" not in rows[0] or value_column not in rows[0]:
        raise ValueError(f"{path} must contain _step and {value_column} columns")
    return [int(row["_step"]) for row in rows], [float(row[value_column]) for row in rows]


def require_rounded_metric_match(
    csv_values: list[float],
    logged_values: list[float],
    description: str,
) -> None:
    """Verify that full-precision CSV values match four-decimal log output."""
    if len(csv_values) != len(logged_values):
        raise ValueError(f"{description} CSV and raw log have different row counts")
    for index, (csv_value, logged_value) in enumerate(zip(csv_values, logged_values, strict=True)):
        if f"{csv_value:.4f}" != f"{logged_value:.4f}":
            raise ValueError(
                f"{description} CSV and raw log differ at row {index}: "
                f"{csv_value} != {logged_value} (four-decimal log precision)"
            )


def read_git_head(directory: Path) -> str:
    head_path = directory / "git-head.txt"
    if not head_path.exists():
        raise ValueError(f"Missing required commit record: {head_path}")
    git_head = head_path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", git_head) is None:
        raise ValueError(f"Invalid commit hash in {head_path}: {git_head!r}")
    return git_head


def trimmed(values: list[float], proportion: float = 0.05) -> tuple[list[float], int]:
    ordered = sorted(values)
    trim_count = int(len(ordered) * proportion)
    if not ordered or trim_count * 2 >= len(ordered):
        raise ValueError("Not enough timing windows for the requested trimming")
    if trim_count == 0:
        return ordered, 0
    return ordered[trim_count:-trim_count], trim_count


def checkpoint_accuracy(directory: Path) -> tuple[float, float] | None:
    history_path = directory / "checkpoints" / "checkpoint_history.json"
    if not history_path.exists():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    checkpoints = history.get("checkpoints", [])
    best_checkpoints = history.get("best_checkpoints", [])
    if not checkpoints or not best_checkpoints:
        return None
    final = max(checkpoints, key=lambda item: (item["step"], item["timestamp"]))
    return float(final["metrics"]["val/acc"]), max(float(item["metric_value"]) for item in best_checkpoints)


def wall_time_seconds(directory: Path) -> float | None:
    start_path = directory / "start.txt"
    end_path = directory / "end.txt"
    if not start_path.exists() or not end_path.exists():
        return None
    start = datetime.fromisoformat(start_path.read_text(encoding="utf-8").strip())
    end = datetime.fromisoformat(end_path.read_text(encoding="utf-8").strip())
    elapsed = (end - start).total_seconds()
    if elapsed < 0:
        raise ValueError(f"Run end precedes run start in {directory}")
    return elapsed


def logged_train_eval_seconds(text: str) -> float:
    start_match = require_match(TRAIN_START_TIME_RE, text, "training start time")
    end_matches = list(TEST_ACC_TIME_RE.finditer(text))
    if not end_matches:
        raise ValueError("Could not find test completion time in the raw log")
    start = datetime.fromisoformat(start_match.group(1).replace(",", "."))
    end = datetime.fromisoformat(end_matches[-1].group(1).replace(",", "."))
    elapsed = (end - start).total_seconds()
    if elapsed < 0:
        raise ValueError("Test completion precedes training start in the raw log")
    return elapsed


def parse_run(directory: Path, label: str) -> dict[str, Any]:
    log_path = find_one(directory, "*.log", "raw log")
    train_csv = find_one(directory, "*_train.csv", "training CSV")
    val_csv = find_one(directory, "*_val.csv", "validation CSV")
    text = log_path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")

    windows = [
        {
            "step": int(step),
            "loss": float(loss),
            "model_step_time_s": float(step_time),
            "seed_nodes_per_s": float(throughput),
        }
        for step, loss, step_time, throughput in STEP_RE.findall(text)
    ]
    if not windows:
        raise ValueError(f"No periodic timing windows found in {log_path}")

    epoch_rows = [
        {
            "epoch": index,
            "step": int(step),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        for index, (step, train_loss, val_loss, val_acc) in enumerate(EPOCH_RE.findall(text), start=1)
    ]
    if not epoch_rows:
        raise ValueError(f"No epoch summaries found in {log_path}")

    initial_match = require_match(INITIAL_RE, text, "initial validation metrics")
    train_steps, train_losses = read_metric_csv(train_csv, "train/loss")
    val_steps, val_losses = read_metric_csv(val_csv, "val/loss")
    if train_steps != [window["step"] for window in windows]:
        raise ValueError(f"Training CSV steps do not match timing-window steps in {log_path}")
    expected_val_steps = [0] + [row["step"] for row in epoch_rows]
    if val_steps != expected_val_steps:
        raise ValueError(f"Validation CSV steps do not match epoch steps in {log_path}")

    require_rounded_metric_match(
        train_losses,
        [float(window["loss"]) for window in windows],
        "Training loss",
    )
    require_rounded_metric_match(
        val_losses,
        [float(initial_match.group(1))] + [float(row["val_loss"]) for row in epoch_rows],
        "Validation loss",
    )

    initial_acc = float(initial_match.group(2))
    for row, val_loss in zip(epoch_rows, val_losses[1:], strict=True):
        row["val_loss"] = val_loss

    exact_accuracy = checkpoint_accuracy(directory)
    if exact_accuracy is None:
        final_val_acc = float(epoch_rows[-1]["val_acc"])
        best_val_acc = max(float(row["val_acc"]) for row in epoch_rows)
        accuracy_source = "raw log (four-decimal precision)"
    else:
        final_val_acc, best_val_acc = exact_accuracy
        epoch_rows[-1]["val_acc"] = final_val_acc
        accuracy_source = "checkpoint history"

    epoch_end_steps = {int(row["step"]) for row in epoch_rows}
    steady_windows = [window for window in windows if window["step"] >= 100 and window["step"] not in epoch_end_steps]
    steady_times, trim_count = trimmed([float(window["model_step_time_s"]) for window in steady_windows])
    steady_throughput, throughput_trim_count = trimmed([float(window["seed_nodes_per_s"]) for window in steady_windows])
    if trim_count != throughput_trim_count:
        raise ValueError("Timing and throughput series produced different trim counts")

    return {
        "label": label,
        "git_head": read_git_head(directory),
        "artifacts": {
            "log": str(log_path),
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
        },
        "completed_successfully": "python_exit_code=0" in text,
        "wall_time_s": wall_time_seconds(directory),
        "logged_train_eval_time_s": logged_train_eval_seconds(text),
        "first_training_step_time_s": float(require_match(FIRST_STEP_RE, text, "first-step timing").group(1)),
        "steady_timing": {
            "source_windows": len(steady_windows),
            "trimmed_windows": len(steady_times),
            "trim_each_tail": trim_count,
            "mean_model_step_time_s": mean(steady_times),
            "median_model_step_time_s": median(steady_times),
            "mean_seed_nodes_per_s": mean(steady_throughput),
            "median_seed_nodes_per_s": median(steady_throughput),
        },
        "initial": {"val_loss": val_losses[0], "val_acc": initial_acc},
        "epochs": epoch_rows,
        "final_val_acc": final_val_acc,
        "best_val_acc": best_val_acc,
        "validation_accuracy_source": accuracy_source,
        "test_loss": float(require_match(TEST_LOSS_RE, text, "test loss").group(1)),
        "test_acc": float(require_match(TEST_ACC_RE, text, "test accuracy").group(1)),
        "plot_data": {
            "train_steps": train_steps,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
    }


def plot_curves(runs: dict[str, dict[str, Any]], output_path: Path) -> None:
    colors = ["#2563eb", "#f97316"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for (label, run), color in zip(runs.items(), colors, strict=True):
        plot_data = run["plot_data"]
        axes[0].plot(
            plot_data["train_steps"],
            plot_data["train_losses"],
            label=label,
            color=color,
            linewidth=1.5,
        )
        epoch_x = list(range(len(plot_data["val_losses"])))
        val_accuracy = [run["initial"]["val_acc"]] + [row["val_acc"] for row in run["epochs"]]
        axes[1].plot(epoch_x, plot_data["val_losses"], marker="o", label=label, color=color)
        axes[2].plot(epoch_x, val_accuracy, marker="o", label=label, color=color)

    axes[0].set(
        title="Training loss",
        xlabel="Training step",
        ylabel="Cross-entropy loss (log scale)",
    )
    axes[0].set_yscale("log")
    axes[1].set(title="Validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[2].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("GraphSAGE Reddit: matched CPU and TT runs")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    runs = {
        args.cpu_label: parse_run(args.cpu_dir, args.cpu_label),
        args.tt_label: parse_run(args.tt_dir, args.tt_label),
    }
    cpu = runs[args.cpu_label]
    tt = runs[args.tt_label]
    unsuccessful = [label for label, run in runs.items() if not run["completed_successfully"]]
    if unsuccessful:
        raise ValueError(f"Run did not record python_exit_code=0: {', '.join(unsuccessful)}")
    if cpu["git_head"] != tt["git_head"]:
        raise ValueError(f"CPU and TT artifacts use different commits: {cpu['git_head']} != {tt['git_head']}")

    summary = {
        "commit": cpu["git_head"],
        "runs": runs,
        "parity": {
            "final_train_loss_abs_diff": abs(cpu["epochs"][-1]["train_loss"] - tt["epochs"][-1]["train_loss"]),
            "final_val_loss_abs_diff": abs(cpu["epochs"][-1]["val_loss"] - tt["epochs"][-1]["val_loss"]),
            "final_val_acc_abs_diff": abs(cpu["final_val_acc"] - tt["final_val_acc"]),
            "test_loss_abs_diff": abs(cpu["test_loss"] - tt["test_loss"]),
            "test_acc_abs_diff": abs(cpu["test_acc"] - tt["test_acc"]),
        },
        "performance_ratio_cpu_over_tt": {
            "steady_model_step_speedup": (
                tt["steady_timing"]["mean_model_step_time_s"] / cpu["steady_timing"]["mean_model_step_time_s"]
            ),
            "steady_throughput_ratio": (
                cpu["steady_timing"]["mean_seed_nodes_per_s"] / tt["steady_timing"]["mean_seed_nodes_per_s"]
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_curves(runs, args.output_dir / "graphsage-cpu-tt-curves.png")
    for run in runs.values():
        run.pop("plot_data")
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {args.output_dir / 'graphsage-cpu-tt-curves.png'}")


if __name__ == "__main__":
    main()
