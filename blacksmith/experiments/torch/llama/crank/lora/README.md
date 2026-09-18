# Llama with LoRA Experiment in tt-crank

This directory contains the code for the Llama with LoRA fine-tuning experiment run through
tt-crank, the `torch.compile` `tt` backend that ships inside tt-mlir (no tt-xla in the loop).
It mirrors the [TT-XLA LoRA experiment](../../xla/lora/README.md): same datasets, logger,
checkpoints and validation cadence, driven by `train_crank.py` instead of `train.py`.

- Llama 3.1 8B model specification can be found [here](https://huggingface.co/meta-llama/Llama-3.1-8B).

Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).

## Overview

The experiment applies LoRA to a pre-trained Llama model on the SST-2 sentiment analysis dataset
or the Alpaca instruction dataset. Only the base model's weights can be stored in block floating
point (`bfp_bf8`, `bfp_bf4`) while the LoRA adapters and all activations stay in bf16, which is what
the `_bfp8` and `_bfp4` configs exercise.

Every run reports, next to the loss, the step time, tokens per second and two utilization
numbers: MFU from an analytical FLOP estimate and HFU from the exact FLOPs tt-mlir reports for the
compiled graphs. The configs also dump the TTIR and TTNN IR of the first step's graphs under
`.data/artifacts/`.

## Environment

tt-crank is a subproject of tt-mlir and reuses tt-mlir's virtualenv. Build tt-mlir with
`-DTTMLIR_ENABLE_CRANK=ON`, install the Python package with `./tt-crank/scripts/install-py` (both
from the tt-mlir root), then activate that environment from here:

```bash
export TT_MLIR_HOME=/path/to/tt-mlir
source env/activate --mlir
```

The first activation installs the blacksmith-specific packages from `env/mlir_requirements.txt`
into the tt-mlir venv.

## Training

All configs run on a single P150 (Blackhole) chip.

### Llama 3.1 8B Training

```bash
python3 blacksmith/experiments/torch/llama/crank/train_crank.py --config blacksmith/experiments/torch/llama/crank/lora/single_chip/llama_3_1_8b_sst2.yaml
python3 blacksmith/experiments/torch/llama/crank/train_crank.py --config blacksmith/experiments/torch/llama/crank/lora/single_chip/llama_3_1_8b_sst2_bfp8.yaml
python3 blacksmith/experiments/torch/llama/crank/train_crank.py --config blacksmith/experiments/torch/llama/crank/lora/single_chip/llama_3_1_8b_sst2_bfp4.yaml
python3 blacksmith/experiments/torch/llama/crank/train_crank.py --config blacksmith/experiments/torch/llama/crank/lora/single_chip/llama_3_1_8b_alpaca_bfp8.yaml
```

### Training Configurations

| Architecture | Config | Dataset | Weight storage | Method |
| ------------ | ------ | ------- | -------------- | ------ |
| P150 | [llama_3_1_8b_sst2.yaml](single_chip/llama_3_1_8b_sst2.yaml) | SST2 | bf16 | LoRA |
| P150 | [llama_3_1_8b_sst2_bfp8.yaml](single_chip/llama_3_1_8b_sst2_bfp8.yaml) | SST2 | bfp_bf8 | LoRA |
| P150 | [llama_3_1_8b_sst2_bfp4.yaml](single_chip/llama_3_1_8b_sst2_bfp4.yaml) | SST2 | bfp_bf4 | LoRA |
| P150 | [llama_3_1_8b_alpaca_bfp8.yaml](single_chip/llama_3_1_8b_alpaca_bfp8.yaml) | Alpaca | bfp_bf8 | LoRA |

The configs are short runs capped with `max_steps`, matching the tt-crank experiments they
were ported from. Remove `max_steps` to train the full epoch(s).

## Data

See the [TT-XLA LoRA README](../../xla/lora/README.md#data) for the SST-2 description. Alpaca
(`tatsu-lab/alpaca`) is an instruction dataset of 52K instruction/input/output triples; the loader
splits off 2% of it for validation.

## Configuration

The configs share the parameters of the TT-XLA LoRA experiment (see its
[parameter table](../../xla/lora/README.md#configuration-paramaters)) and add the following.

| Parameter | Description | Default Value |
| --------- | ----------- | ------------- |
| `adam_beta1`, `adam_beta2`, `adam_eps` | AdamW moment decays and epsilon. | 0.9, 0.999, 1e-8 |
| `weight_decay` | AdamW decoupled weight decay. | 0.0 |
| `max_steps` | Stop after this many optimizer steps; `null` trains `num_epochs` in full. | `null` |
| `experimental_weight_dtype` | Storage dtype of every matmul weight: `"bfp_bf8"`, `"bfp_bf4"` or `"bf16"`. | `null` (bf16) |
| `opt_level` | tt-mlir optimization level. | 1 |
| `enable_const_eval` | Hoist constant subgraphs out of the per-step graph. | False |
| `enable_trace` | Capture the compiled programs as a device trace. | False |
| `perf_metrics_enabled` | Ask tt-mlir for exact FLOPs per compiled graph (reported as `perf/hfu`). | True |
| `perf_metrics_file` | Base name of the report tt-mlir writes (`<name>.json`). | "perf_metrics" |
| `artifacts_name` | Dump the first step's TTIR/TTNN IR under `<artifacts_dir>/<artifacts_name>_<timestamp>/`; `null` skips. | `null` |
| `artifacts_dir` | Root directory for IR dumps. | ".data/artifacts" |
| `use_tt` | Run on the TT device; `False` runs the same loop eagerly on CPU. | True |
