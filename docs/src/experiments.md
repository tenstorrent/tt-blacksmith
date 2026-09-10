# Experiments

This page provides an overview of the experiments included in this repository, detailing their organization.

## Available Experiments

The following table provides an overview of different model and method combinations within various frameworks explored in this project.

| Framework | Model | Method  | Details |
| --------- | ----- | ------- | ------- |
| PyTorch | MLP | Full-model  | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/mnist/README.md) |
| PyTorch | CNN | Full-model | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/mnist_cnn/README.md) |
| PyTorch | Llama 3.2 1B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/lora/README.md#llama-32-1b-training) |
| PyTorch | Llama 3.2 1B | Adapters | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/adapters/README.md) |
| PyTorch | Llama 3.2 3B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/lora/README.md#llama-32-3b-training) |
| PyTorch | Llama 3.1 8B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/lora/README.md#llama-31-8b-training) |
| PyTorch | Llama 3.1 8B Instruct | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/lora/README.md#llama-31-8b-instruct-training) |
| PyTorch | Llama 3.1 70B | LoRA| [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/lora/README.md#llama-31-70b-training) |
| PyTorch | Llama 3.3 70B Instruct | LoRA| [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/lora/README.md#llama-33-70b-instruct-training) |
| PyTorch | GPT-OSS 20B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/gpt_oss/README.md#gpt-oss-20b-training) |
| PyTorch | GPT-OSS 120B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/gpt_oss/README.md#gpt-oss-120b-training) |
| PyTorch | Qwen 2.5 0.5B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/qwen/README.md#qwen-25-05b-training) |
| PyTorch | Qwen 2.5 1.5B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/qwen/README.md#qwen-25-15b-training) |
| Pytorch | Qwen 3 4B Instruct 2507 | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/qwen/README.md#qwen-3-4b-instruct-2507-training) |
| Pytorch | Qwen 3 8B-Base | LoRA| [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/qwen/README.md#qwen-3-8b-base-training) |
| Pytorch | Qwen 3 8B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/qwen/README.md#qwen-3-8b-training) |
| PyTorch | Qwen 3 32B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/qwen/README.md#qwen-3-32b-training) |
| PyTorch | Gemma 3 1B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/gemma/README.md) |
| PyTorch | Gemma 1.1 2B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/gemma11/lora/README.md) |
| PyTorch | Gemma 1.1 2B | LoRA, DPO | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/gemma11/dpo/README.md) |
| PyTorch | Gemma 2 2B | LoRA, GRPO | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/gemma2/grpo/README.md) |
| PyTorch | Gemma 4 E2B Instruct | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/gemma4/README.md) |
| PyTorch | ALBERT | Adapters | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/albert/README.md) |
| PyTorch | Phi-1 | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/phi/README.md#phi1) |
| PyTorch | Phi-1.5 | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/phi/README.md#phi-15) |
| PyTorch | GATv2 | Full-model | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/BOUNTIES/gatv2_pubmed/README.md) |
| PyTorch | GraphSAGE | Full-model | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/BOUNTIES/graphsage_reddit/README.md) |
| PyTorch | Wan 2.2 5b | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/wan2_2/README.md) |
| JAX | MLP | Full-model | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/mnist/README.md) |
| JAX | NeRF | Full-model | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/nerf/README.md) |
| JAX | Llama 3.2 1B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/llama/lora/README.md) |
| JAX | Llama 3.2 1B | DoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/llama/dora/README.md) |
| JAX | DistilBERT | Distillation | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/distil_bert/README.md) |
| EasyDel | Qwen 3 0.6B | LoRA | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/easydel/qwen/lora/README.md) |
| Lightning | NeRF | Full-model | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/lightning/nerf/README.md) |


## Navigating the Experiment Structure
Within this repository, you'll find the following structure to help you navigate the experimental setup:

- `datasets/`: The dataset loaders for specific model training are defined in this directory and organized by the framework they utilize. For example, the loader for the MNIST dataset in PyTorch can be found at `datasets/torch/mnist/`.
- `models/`: This directory is organized by framework. Within it, you'll find subdirectories (e.g., `jax/`, `torch/`) containing the model implementations or loader scripts specific to that framework. For instance, the PyTorch implementation of a model for MNIST training would be located in `models/torch/mnist/`.
- `experiments/`: Experiments are organized first by the framework they utilize, and then by the specific model or task. For example, the PyTorch-based MNIST experiment can be found under `experiments/torch/mnist/`. Within each experiment directory, you will typically find the following subdirectories and files:

    - Subdirectories named after fine-tuning methods used in experiments (e.g. `lora`, `dpo`, `adapters`).
    - Subdirectories specifying the compute environment (e.g. `single_chip`, `quietbox`, `loudbox`, `galaxy`).
    - Within these subdirectories there are YAML files containing the specific configuration parameters, named after the model and dataset used (e.g. `gemma11_sst2.yaml` - the full path in the `experiments` directory for this file is `torch/gemma11/lora/single_chip/gemma11_sst2.yaml`).
    - A Python file defining the configuration structure for the experiment (e.g. `configs.py`).
    - A Python training script (`train.py`) responsible for running the experiment using the defined configurations.
    - A README file listing all experiments with the given model and fine-tuning method.
