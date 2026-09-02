# GraphSAGE / Reddit - CPU vs TT results

These results were collected from commit
`f353f935d3a5e95cf0a0ad08993e24cbe0973b46` on September 2, 2026. The two
full runs used the same scatter-free SpMM model, sampling configuration,
static shapes, optimizer hyperparameters, seed, and five-epoch schedule. Only
the execution backend and run/output metadata differed:

- **CPU parity:** SpMM GraphSAGE on the host CPU.
- **TT:** the same model on a single Wormhole N150 through TT-XLA (`xla:0`).

The full TT run used an N150 because an N300 cloud allocation was not
available. N300 coverage is limited to the maintainer CI smoke run described
below, so none of the full-run numbers in this file should be read as N300
results.

## Run configuration

| Setting | Value |
|---|---:|
| Dataset | Reddit (232,965 nodes, 114,615,892 edges) |
| Model parameters | 329,513 |
| Hidden channels | 256 |
| Batch / validation batch size | 32 / 32 |
| Neighbor fanouts | `[5, 3]` |
| Static node / edge capacity | 769 / 736 |
| Dropout | 0.0 |
| Learning rate / weight decay | 0.001 / 0.0005 |
| Epochs / training steps | 5 / 23,975 |
| Seed / deterministic | 42 / yes |

The runs used Ubuntu 22.04.5, Python 3.12.0, PyTorch 2.11.0+cpu,
torch-xla 2.9.0+gited8a445, and PyTorch Geometric 2.8.0.post1. The CPU host
had two AMD EPYC 7352 sockets (96 logical CPUs).

## Accuracy parity

| Metric | CPU (SpMM) | Wormhole N150 | Absolute difference |
|---|---:|---:|---:|
| Final-epoch mean train loss | 0.2797 | 0.2777 | 0.0020 |
| Final validation loss | 0.280388 | 0.282091 | 0.001703 |
| Final validation accuracy | 0.927028 | 0.925265 | 0.001762 |
| Best validation accuracy | 0.927028 | 0.925517 | 0.001511 |
| Test loss | 0.294802 | 0.298679 | 0.003877 |
| **Test accuracy** | **0.924744** | **0.923702** | **0.001041** |

The final test-accuracy gap is about **0.10 percentage points**. Training loss,
validation loss, and validation accuracy also track closely throughout the
five epochs. The CPU/N150 comparison plot was generated from the recorded CSV
and raw-log artifacts; its training-loss panel uses a logarithmic y-axis.

## Timing

The model-step timer starts after neighbor sampling and static batch
preparation. It covers `zero_grad`, model forward, masked loss, backward, and
the synchronized optimizer step. It therefore measures the training work sent
through the model path, but excludes the CPU-side `NeighborLoader` and batch
preparation.

| Metric | CPU (SpMM) | Wormhole N150 |
|---|---:|---:|
| Steady model-step time | 11.364 ms | 59.181 ms |
| Steady seed-node throughput | 2,817.5 nodes/s | 542.4 nodes/s |
| Logged train/eval interval | about 6m 33s | about 28m 29s |

Steady values are 5%-trimmed means of the periodic timing windows logged at
step 100 or later, after excluding the step-50 warm-up window and the epoch-end
remainder windows, each of which contains the final partial batch. On this
small, statically padded sampled-graph workload, the host CPU model-step metric
is about 5.2x faster; these results do not claim a TT speedup.

The initial N150 smoke run measured **20.7714 seconds** for its cold-cache
first-training-step metric, including compilation triggered by that step.
Initial validation had already compiled the forward-only path, so this is not
the workload's total compilation time. The full run reused the TT-Metal JIT
kernel cache (`388/388` cache hits), so its first training step was a warm-cache
measurement of 0.7221 seconds rather than a second cold-cache result.

## N300 CI status

The isolated [N300/PyG maintainer CI job][n300-ci] completed successfully: the
GraphSAGE hardware smoke case and 18 focused tests all passed (`19 passed` in
123.68 seconds). The smoke overlay limits initial validation, training,
post-training validation, and testing to two batches each. It verifies N300
execution and CI integration, but it is not a full convergence or performance
run.

## Limitations

- Neighbor sampling and batch preparation remain on CPU. Model forward,
  backward, and the optimizer step are the TT workload.
- The TT path uses fixed sampled-graph capacities and a scatter-free SpMM
  reformulation to keep compilation reusable.
- XLA SpMM aggregation uses bfloat16 matmul inputs with float32 accumulation;
  the CPU parity path keeps float32 inputs. The accuracy comparison therefore
  includes this expected precision difference.
- Adam uses the same learning rate and weight decay on both backends, with
  `capturable` enabled only for TT execution.
- Wormhole emitted the known HiFi4/fp32-accumulation accuracy warning. The
  observed CPU/N150 accuracy gap remained about 0.10 percentage points.
- Full N300 convergence and performance results remain the next hardware step
  if an N300 allocation becomes available.

## Run the workload

```bash
source env/activate --xla
pip install -r blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/requirements.txt

# Matched CPU parity run
python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_spmm_cpu.yaml

# Matched TT run (N150 results above; the config also targets one Wormhole chip)
python3 blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/train.py \
  --config blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/single_chip/graphsage_reddit_tt.yaml
```

Download and process Reddit before collecting timings so dataset setup is not
part of the comparison. Run the CPU and TT commands sequentially on the same
software stack.

[n300-ci]: https://github.com/tenstorrent/tt-blacksmith/actions/runs/32417455921/job/97565328507
