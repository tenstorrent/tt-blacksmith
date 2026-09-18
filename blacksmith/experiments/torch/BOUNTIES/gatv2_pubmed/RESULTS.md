# GATv2 / PubMed — CPU vs TT results

Full transductive training on PubMed (19,717 nodes, 88,648 edges; 108,365 with
self-loops), seed 42, identical hyperparameters, all on tt-xla `1.3.0.dev`. Two
runs isolate the backend:

- **CPU (stock `GATv2Conv`)** — original scatter model, on CPU.
- **TT (`SpMMGATv2Conv`)** — the matmul rewrite (`use_spmm: true`), on a Wormhole
  N300 (single chip).

The SpMM layer is bit-equivalent to `GATv2Conv` per step (loss identical,
per-parameter gradient cosine 1.0), so the **rewrite** preserves the math exactly;
CPU-stock vs TT-SpMM then isolates the **backend** (CPU vs device).

## Accuracy parity

| Metric | CPU (stock GATv2Conv) | TT (SpMM) |
|---|---|---|
| Device | cpu | Wormhole N300, single chip (`xla:0`) |
| Convolution | `GATv2Conv` (scatter) | `SpMMGATv2Conv` (matmul) |
| **Test accuracy** | **0.7800** | **0.7840** |
| Best val accuracy | 0.7920 | 0.8000 |
| Test loss | 0.5654 | 0.5548 |
| Final train loss | 0.0538 | 0.0438 |
| Epochs (early stop, patience 50) | 106 | 122 |

TT-SpMM lands at **0.7840** vs the CPU baseline's **0.7800** (+0.0040), within
run-to-run noise: the rewrite is bit-equivalent per step, so this is purely CPU
vs device.

## Curves

Loss and validation-accuracy curves for both runs track each other closely
throughout training (TT tracks the CPU baseline within noise). The comparison
plots are attached to the pull request description. Per-run plots are also
written to `results/` by `train.py`.

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

# TT (SpMM)
python3 blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed/train.py --config $CFG/gatv2_pubmed_tt.yaml
```

Each run writes `results/results_summary.json` (final metrics) and per-run
loss/accuracy plots; the comparison plots are built from the run logs.
