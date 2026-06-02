# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GraphSAGE Reddit — training analysis.

Run from the tt-blacksmith root:
    python blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/analyze.py

Outputs (saved to graphsage_reddit/plots/):
    training_curves.png  — train loss, val loss, val accuracy per epoch
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_LOG = Path("/tmp/graphsage_reddit_train.log")

EXPERIMENT_DIR = Path(__file__).parent
PLOTS_DIR = EXPERIMENT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Parse training log — per-epoch metrics
# ──────────────────────────────────────────────────────────────
epochs = []
train_loss = []
val_loss = []
val_acc = []

if not TRAIN_LOG.exists():
    raise FileNotFoundError(f"Training log not found: {TRAIN_LOG}")

pattern = re.compile(
    r"Epoch (\d+)/\d+ \| train_loss=([\d.]+)\s+val_loss=([\d.]+)\s+val_acc=([\d.]+)"
)
seen = set()
for line in TRAIN_LOG.read_text().splitlines():
    m = pattern.search(line)
    if m:
        ep = int(m.group(1))
        if ep not in seen:
            seen.add(ep)
            epochs.append(ep)
            train_loss.append(float(m.group(2)))
            val_loss.append(float(m.group(3)))
            val_acc.append(float(m.group(4)))

print(f"Parsed {len(epochs)} epochs from training log")

best_val_acc   = max(val_acc)
best_epoch     = epochs[val_acc.index(best_val_acc)]
min_val_loss   = min(val_loss)
min_loss_epoch = epochs[val_loss.index(min_val_loss)]

# ──────────────────────────────────────────────────────────────
# Plot: Training curves
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
fig.suptitle(
    f"GraphSAGE on Reddit — Training History ({max(epochs)} epochs)",
    fontsize=13, fontweight="bold",
)

ax = axes[0]
ax.plot(epochs, train_loss, color="steelblue", linewidth=2, label="Train loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Training Loss")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(epochs, val_loss, color="tomato", linewidth=2, label="Val loss")
ax.scatter([min_loss_epoch], [min_val_loss], color="darkred", zorder=5, s=80,
           label=f"Min {min_val_loss:.4f} @ ep {min_loss_epoch}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Validation Loss")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

ax = axes[2]
ax.plot(epochs, [a * 100 for a in val_acc], color="darkorange",
        linewidth=2, label="Val acc")
ax.scatter([best_epoch], [best_val_acc * 100], color="red", zorder=5, s=80,
           label=f"Best {best_val_acc*100:.2f}% @ ep {best_epoch}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Validation Accuracy")
ax.set_ylim(93, 97.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
p = PLOTS_DIR / "training_curves.png"
plt.savefig(p, dpi=150)
plt.close()
print(f"Saved: {p}")

print(f"\nBest val acc: {best_val_acc*100:.2f}% @ epoch {best_epoch}")
print(f"Min val loss: {min_val_loss:.4f} @ epoch {min_loss_epoch}")
