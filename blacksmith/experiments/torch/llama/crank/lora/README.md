# Llama with QLoRA Experiment in TT-Crank

This directory contains the code for the Llama with modified QLoRA fine-tuning experiment run through tt-crank.

- Llama 3.1 8B model specification can be found [here](https://huggingface.co/meta-llama/Llama-3.1-8B).

Original LoRA paper can be found [here](https://arxiv.org/pdf/2106.09685).
The QLoRA paper can be found [here](https://arxiv.org/abs/2305.14314).

## Overview

The experiment applies a modified QLoRA to a pre-trained Llama model on the SST-2 sentiment analysis dataset
or the Alpaca instruction dataset. The base model's weights are stored in block floating
point (`bfp_bf8`, `bfp_bf4`) while the LoRA adapters and all activations stay in bf16, as opposed to the QLoRA
quantization scheme, which packs the frozen weights into 4-bit NormalFloat with double-quantized scales and
dequantizes them back to bf16 on the fly before every matmul. Block floating point is a native Tenstorrent
format: each block of 16 values shares one exponent and keeps an 8- or 4-bit mantissa, and the matmul kernels
consume it directly.

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

Until this [issue](https://github.com/tenstorrent/tt-metal/issues/56613) is resolved, use the following flag to avoid hanging.

```bash
export TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=536870912
```

## Training

### Llama 3.1 8B Training

```bash
python3 blacksmith/experiments/torch/llama/crank/train.py --config blacksmith/experiments/torch/llama/crank/lora/single_chip/llama_3_1_8b_sst2.yaml
python3 blacksmith/experiments/torch/llama/crank/train.py --config blacksmith/experiments/torch/llama/crank/lora/single_chip/llama_3_1_8b_sst2_bfp8.yaml
python3 blacksmith/experiments/torch/llama/crank/train.py --config blacksmith/experiments/torch/llama/crank/lora/single_chip/llama_3_1_8b_sst2_bfp4.yaml
python3 blacksmith/experiments/torch/llama/crank/train.py --config blacksmith/experiments/torch/llama/crank/lora/single_chip/llama_3_1_8b_alpaca.yaml
```

### Training Configurations

| Architecture | Config | Dataset | Weight storage | Method |
| ------------ | ------ | ------- | -------------- | ------ |
| P150 | [llama_3_1_8b_sst2.yaml](single_chip/llama_3_1_8b_sst2.yaml) | SST2 | bf16 | LoRA |
| P150 | [llama_3_1_8b_sst2_bfp8.yaml](single_chip/llama_3_1_8b_sst2_bfp8.yaml) | SST2 | bfp_bf8 | LoRA |
| P150 | [llama_3_1_8b_sst2_bfp4.yaml](single_chip/llama_3_1_8b_sst2_bfp4.yaml) | SST2 | bfp_bf4 | LoRA |
| P150 | [llama_3_1_8b_alpaca_bfp8.yaml](single_chip/llama_3_1_8b_alpaca_bfp8.yaml) | Alpaca | bfp_bf8 | LoRA |


## Data

GLUE, the General Language Understanding Evaluation benchmark (https://gluebenchmark.com/) is a collection of resources for training, evaluating, and analyzing natural language understanding systems.
The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. It uses the two-way (positive/negative) class split, with only sentence-level labels.
Each example consists of a sentence from movie reviews labeled as either positive or negative sentiment.
This dataset is commonly used to evaluate the performance of natural language understanding models on sentiment analysis tasks.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/nyu-mll/glue)

Example
```
{
  "sentence": "A touching and insightful film.",
  "label": 1
}
```
- sentence: A short movie review or phrase.
- label: Sentiment label (1 for positive, 0 for negative).

Alpaca (`tatsu-lab/alpaca`, https://crfm.stanford.edu/2023/03/13/alpaca.html) is an instruction-following dataset of 52K examples released by Stanford CRFM.
The examples were generated with OpenAI's text-davinci-003 using the Self-Instruct method, seeded from 175 human-written instruction/output pairs.
Each example consists of an instruction describing a task, an optional input providing context for that task, and the expected output; roughly 40% of the examples include a non-empty input.
The tasks span open-ended generation, brainstorming, classification, rewriting, question answering, and simple reasoning.
This dataset is commonly used for supervised fine-tuning of base language models into instruction-following assistants.

Source: [Hugging Face Dataset Hub](https://huggingface.co/datasets/tatsu-lab/alpaca)

Example

{
"instruction": "Give three tips for staying healthy.",
"input": "",
"output": "1. Eat a balanced diet and make sure to include plenty of fruits and vegetables. 2. Exercise regularly to keep your body active and strong. 3. Get enough sleep and maintain a consistent sleep schedule."
}

- instruction: A natural language description of the task to be performed.
- input: Optional additional context or input for the task; empty string when not applicable.
- output: The expected response generated by text-davinci-003 for the given instruction (and input, if any).

The loader formats each example with the standard Alpaca prompt template and splits off 2% of the data for validation.

## Configuration

| Parameter | Description | Default Value |
| --------- | ----------- | ------------- |
| `dataset_id` | The dataset used for fine-tuning (`"sst2"` or `"alpaca"`). | "sst2" |
| `model_name` | Name or path of the pre-trained model. | "meta-llama/Llama-3.1-8B" |
| `max_length` | Maximum token length for inputs. | 512 |
| `dtype` | Data type used during training. | "torch.bfloat16" |
| `experimental_weight_dtype` | Storage dtype of every matmul weight: `"bfp_bf8"`, `"bfp_bf4"` or `"bf16"`. | "bf16" |
| `training_model_type` | Which type of finetuning to do. | "lora" |
| `learning_rate` | Learning rate for the optimizer. | 1e-4 |
| `adam_beta1`, `adam_beta2`, `adam_eps` | AdamW moment decays and epsilon. | 0.9, 0.999, 1e-8 |
| `weight_decay` | AdamW decoupled weight decay. | 0.01 |
| `batch_size` | Number of samples per training batch. | 2 |
| `gradient_accumulation_steps` | Steps to accumulate gradients before updating (Alpaca config only). | 1 |
| `num_epochs` | Total number of training epochs. | 1 |
| `max_steps` | Stop after this many optimizer steps; `null` trains `num_epochs` in full. | 100 |
| `ignored_index` | Label value that marks tokens excluded from the loss. | -100 |
| `lora_r` | Rank of LoRA adaptation matrices. | 4 |
| `lora_alpha` | Scaling factor for LoRA updates. | 8 |
| `lora_target_modules` | Target modules for LoRA adaptation. | ["q_proj", "v_proj"] |
| `lora_task_type` | Training task type for LoRA. | "CAUSAL_LM" |
| `seed` | Random seed for reproducibility. | 23 |
| `deterministic` | Whether to enforce deterministic behavior. | False |
| `log_level` | Logging verbosity level. | "INFO" |
| `use_wandb` | Whether to enable Weights & Biases logging. | True |
| `wandb_project` | Project name for Weights & Biases logging. | "llama8b-p150-lora-crank" |
| `wandb_run_name` | Run name for Weights & Biases tracking. | "tt-llama8b-crank-sst2-bf16" |
| `wandb_tags` | List of tags assigned to the W&B run. | ["test", "crank", "bf16"] |
| `wandb_watch_mode` | Watch mode for model parameter logging. | "all" |
| `wandb_log_freq` | Frequency of logging to Weights & Biases (in steps). | 1000 |
| `model_to_wandb` | Whether to store model checkpoint in Weights & Biases. | False |
| `steps_freq` | Frequency (in steps) for performing periodic actions. | 1 |
| `val_steps_freq` | Frequency (in steps) for performing validation actions. | 20 |
| `epoch_freq` | Frequency (in epochs) for performing periodic actions. | 1 |
| `measure_e2e_time` | Whether to report end-to-end wall-clock time of the run. | True |
| `resume_from_checkpoint` | Whether to resume training from a previous checkpoint. | False |
| `resume_option` | Resume method (`last`, `best`, or `path`). | "last" |
| `checkpoint_path` | Path to a checkpoint if `resume_option="path"`. | "" |
| `checkpoint_metric` | Metric used to rank checkpoints for `best`. | "eval/loss" |
| `checkpoint_metric_mode` | Whether a lower (`min`) or higher (`max`) metric is better. | "min" |
| `keep_last_n` | Number of most recent checkpoints to keep. | 2 |
| `keep_best_n` | Number of best checkpoints to keep. | 1 |
| `save_strategy` | Strategy for saving checkpoints (`epoch`, `step` or `none`). | "none" |
| `project_dir` | Directory for experiment outputs. | "blacksmith/experiments/torch/llama/crank/lora" |
| `save_optim` | Whether to save optimizer state. | False |
| `storage_backend` | Storage backend for saving checkpoints. | "local" |
| `sync_to_storage` | Whether to sync checkpoints to remote storage. | False |
| `load_from_storage` | Whether to load checkpoints from remote storage. | False |
| `remote_path` | Remote storage path (if applicable). | "" |
| `opt_level` | tt-mlir optimization level. | 1 |
| `enable_const_eval` | Hoist constant subgraphs out of the per-step graph. | False |
| `enable_trace` | Capture the compiled programs as a device trace. | False |
| `perf_metrics_enabled` | Ask tt-mlir for exact FLOPs per compiled graph (reported as `perf/hfu`). | True |
| `perf_metrics_file` | Base name of the report tt-mlir writes (`<name>.json`). | "perf_metrics" |
| `artifacts_name` | Dump the first step's TTIR/TTNN IR under `<artifacts_dir>/<artifacts_name>_<timestamp>/`; `null` skips. | "llama_3_1_8b_sst2_bf16" |
| `artifacts_dir` | Root directory for IR dumps. | ".data/artifacts" |
| `framework` | Training framework. | "pytorch" |
| `print_examples` | Log decoded predictions next to their targets during evaluation. | False |
| `use_tt` | Run on the TT device; `False` runs the same loop eagerly on CPU. | True |
