# GATv2 on PubMed (CPU Baseline for Bounty #453)

This directory provides the **PR-1 CPU baseline** requested in issue #453:

- End-to-end GATv2 node classification training on PubMed
- Deterministic training setup (seeded)
- Saved metrics and comparison-ready curves

The **TT-N150/Koyeb execution path** (with targeted CPU fallback and TTIR evidence) is intended for **PR-2**.

## Files

- `test_gatv2_training.py`: CPU training entrypoint
- `test_gatv2_training.yaml`: default config
- `configs.py`: config schema

## Environment (CPU)

Install Python dependencies in your environment:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric matplotlib pyyaml pydantic
```

## Run

```bash
python3 blacksmith/experiments/torch/BOUNTIES/gatv2-pubmed/test_gatv2_training.py \
  --config blacksmith/experiments/torch/BOUNTIES/gatv2-pubmed/test_gatv2_training.yaml
```

## Outputs

By default, outputs are written to:

`blacksmith/experiments/torch/BOUNTIES/gatv2-pubmed/results`

Artifacts:

- `metrics_cpu.csv` - epoch metrics (train/val/test loss and accuracy)
- `summary_cpu.json` - best epoch summary
- `loss_curves_cpu.png` - train/val loss curves
- `accuracy_curves_cpu.png` - train/val/test accuracy curves

## Notes for PR-2 (TT-N150 on Koyeb)

TT execution will reuse the same training loop and metrics contract from this baseline while adding:

1. `use_tt: true` device path on Koyeb N150
2. Function-level fallback wrappers for unsupported ops
3. `TTXLA_LOGGER_LEVEL=DEBUG` validation and TTIR evidence
4. CPU-vs-TT parity table for loss/accuracy trajectories

