# Bounty: Add GATv2 Single-Chip Training on TT-N150 with CPU Baseline

## Background

TT-Blacksmith is a growing collection of model demos showcasing the capabilities of AI models running on Tenstorrent hardware.

This bounty aims to bring up full training of **GATv2** (Graph Attention Network v2) for node classification on the **PubMed citation network dataset** on a single Tenstorrent N150 chip, faithfully reproducing the training setup in PyTorch.

To establish a baseline, the same training must also be run on CPU and the results compared, ensuring that metrics (e.g., loss curves, validation accuracy, test accuracy) align between CPU and TT-N150 runs.

When encountering compilation or runtime issues, contributors are expected to:

- Delegate the issue with a minimal repro to the appropriate Tenstorrent repository (`tt-xla`/`tt-mlir` for compilation, `tt-metal` for runtime).
- Provide a CPU fallback workaround in their training code so training can continue while the issue is investigated.

This bounty is an opportunity for AI developers to:

- Contribute to Tenstorrent's model demos
- Validate single-chip GNN training
- Earn rewards

## Requirements

### Framework

Implementation must be in **PyTorch**.

### GATv2 Training on TT-N150

- Implement end-to-end GATv2 training for node classification on a single chip.
- Train on the **PubMed citation network dataset**.
- Reproduce all training hyperparameters, data setup, and workflow from standard GATv2 implementations.

### CPU Baseline

- Run the same training on CPU.
- Compare results, ensuring parity in metrics and convergence.

### Fallback Mechanism

- In cases where an operation fails on TT-N150, redirect only the minimal possible function containing that operation to CPU.
- **Do not** fallback the entire training step to CPU — training must primarily execute on TT hardware.

### Verification

To confirm training is running on TT hardware, set:

```bash
export TTXLA_LOGGER_LEVEL=DEBUG
```

Look for printed TTIR graphs (e.g., `graph module @jit_training_step`) confirming TT execution.

### Koyeb Machines

- Training should be run on Koyeb-hosted N150 instances.
- Request access here: https://www.koyeb.com/solutions/tenstorrent
- Wait for the onboarding email from Koyeb with setup instructions.

### Documentation

Provide a detailed `README.md` including:

- Setup instructions for both CPU and TT-N150
- Explanation of GATv2 model and dataset setup
- Explanation of fallback logic

### Examples

Include sample input/output and plots of loss/accuracy curves for CPU vs TT-N150 runs.

### Dependencies

Clearly document installation steps and dependencies.

## Contribution Guidelines

1. Fork the `tt-blacksmith` repository.
2. Create a directory:

   ```
   tt-blacksmith/blacksmith/experiments/torch/BOUNTIES/gatv2-pubmed
   ```

3. Add your model implementation, training scripts, and documentation there.
4. Submit **2 pull requests**:
   - **PR 1 — CPU baseline:** A working training example on CPU that demonstrates correct training (loss going down, episode rewards improving, etc.).
   - **PR 2 — TT device execution:** Move training to TT device execution, including fallback logic and CPU vs TT-N150 comparison.
5. Each PR should include:
   - A detailed description
   - Results and comparison plots
   - Relevant information to help reviewers evaluate your contribution

## Evaluation Criteria

| Criterion | Description |
|---|---|
| **Framework Compliance** | Implementation must be in PyTorch. |
| **Hardware Requirement** | Must demonstrate training on a single TT-N150 chip (via Koyeb). |
| **Completeness** | End-to-end GATv2 training on PubMed must be functional. |
| **Metric Parity** | CPU vs TT-N150 runs should show comparable convergence (loss, accuracy, etc.). |
| **Fallback Implementation** | Only targeted CPU fallbacks are acceptable — the full training step must not run on CPU. |
| **Verification** | Use `TTXLA_LOGGER_LEVEL=DEBUG` to confirm TTIR execution of key training functions. |
| **Minimal Repros** | Any delegated issue must include a minimal reproducer. |
| **Code Quality** | Clear, maintainable, and well-commented code. |
| **Documentation** | Setup and usage must be clearly explained, including GATv2 and dataset setup. |
