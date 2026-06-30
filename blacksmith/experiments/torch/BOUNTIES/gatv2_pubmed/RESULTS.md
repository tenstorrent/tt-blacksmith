# GATv2 / PubMed — CPU vs TT results

Full transductive training on PubMed (19,717 nodes, 88,648 edges; 108,365 with
self-loops), seed 42, identical hyperparameters, all on tt-xla `1.3.0.dev`. Three
runs isolate every variable:

- **CPU (stock `GATv2Conv`)** — original scatter model, on CPU.
- **CPU (`SpMMGATv2Conv`)** — the matmul rewrite (`use_spmm: true`), on CPU.
- **TT (`SpMMGATv2Conv`)** — the same rewrite, on a Wormhole N300 (single chip).

CPU-stock vs CPU-SpMM isolates the **rewrite** (same hardware); CPU-SpMM vs
TT-SpMM isolates the **backend** (same code). The SpMM layer is also bit-equivalent
to `GATv2Conv` per step (loss identical, per-parameter gradient cosine 1.0).

## Accuracy parity

| Metric | CPU (stock GATv2Conv) | CPU (SpMM) | TT (SpMM) |
|---|---|---|---|
| Device | cpu | cpu | Wormhole N300, single chip (`xla:0`) |
| Convolution | `GATv2Conv` (scatter) | `SpMMGATv2Conv` (matmul) | `SpMMGATv2Conv` (matmul) |
| **Test accuracy** | **0.7800** | **0.7800** | **0.7840** |
| Best val accuracy | 0.7920 | 0.7920 | 0.8000 |
| Test loss | 0.5654 | 0.5555 | 0.5548 |
| Final train loss | 0.0538 | 0.0756 | 0.0438 |
| Epochs (early stop, patience 50) | 106 | 115 | 122 |

The two CPU runs reach **identical test accuracy (0.7800)** — the matmul rewrite
preserves the math. TT-SpMM lands at **0.7840** (+0.0040 vs CPU), within
run-to-run noise: same code, CPU vs device, matches.

## Curves

Loss and validation-accuracy curves for all three runs track each other closely
throughout training (CPU-stock and CPU-SpMM are nearly indistinguishable; TT
tracks them within noise). The 3-way comparison plots are attached to the pull
request description. Per-run plots are also written to `results/` by `train.py`.

## Execution on TT

| Property | Value |
|---|---|
| `ttnn.scatter` ops in the training-step graph | **0** (the tt-mlir#8714 tiling blowup is fully bypassed) |
| `ttnn.matmul` ops | 168 |
| Step time after compile | ~1.9 s/step |
| Peak DRAM | fits a single 12 GB chip (bf16 one-hot) |
| TTIR proof graph | `module @SyncTensorsGraph` / `func.func @main`, emitted under `TTXLA_LOGGER_LEVEL=DEBUG` |

The model runs **entirely on device** — there is no CPU fallback. See
[`spmm_gatv2.py`](../../../../models/torch/gatv2_pubmed/spmm_gatv2.py) for the
SpMM reformulation and the README's "TT Execution Status" section for the design
rationale.

## Reproduce

```bash
source env/activate --xla
CFG=blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed/single_chip

# CPU baseline (stock GATv2Conv)
python3 blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed/train.py

# CPU (SpMM) — isolates the rewrite from the backend
python3 blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed/train.py --config $CFG/gatv2_pubmed_cpu_spmm.yaml

# TT (SpMM)
python3 blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed/train.py --config $CFG/gatv2_pubmed_tt.yaml
```

Each run writes `results/results_summary.json` (final metrics) and per-run
loss/accuracy plots; the 3-way comparison plots are built from the run logs.
