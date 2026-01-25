# Experiments

This page provides an overview of the experiments included in this repository, detailing their organization.

## Available Experiments

The following table provides an overview of different model and dataset combinations within various frameworks explored in this project.

| Framework | Model | Dataset | Method | Devices | Details |
| --- | --- | --- | --- | --- | --- |
| PyTorch | MLP | MNIST | Full-model | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/mnist/README.md) |
| PyTorch | CNN | MNIST | Full-model | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/mnist/README.md) |
| PyTorch | MLP | MNIST | Full-model, Data parallel | N300 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/mnist/README.md) |
| PyTorch | MLP | MNIST | Full-model, Tensor parallel | N300 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/mnist/README.md) |
| PyTorch | Llama 3.2 1B | SST-2 | LoRA | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/lora/README.md) |
| PyTorch | Llama 3.2 1B | SST-2 | LoRA, Data + Tensor parallel| T3K | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/lora/README.md) |
| PyTorch | Llama 3.2 1B | SST-2 | LoRA, Data + Tensor parallel| WH/BH Galaxy | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/lora/README.md) |
| PyTorch | Llama 3.1 8B | SST-2 | LoRA, Data + Tensor parallel| T3K | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/lora/README.md) |
| PyTorch | Llama 3.1 8B | SST-2 | LoRA, Data + Tensor parallel| BH Galaxy | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/lora/README.md) |
| PyTorch | Llama 3.2 1B | SST-2 | Adapters | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/adapters/README.md) |
| PyTorch | Qwen 2.5 0.5B | Text-to-SQL | LoRA | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/qwen/README.md) |
| PyTorch | Qwen 2.5 1.5B | Text-to-SQL | LoRA | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/qwen/README.md) |
| PyTorch | Gemma 3 1B | SST-2 | LoRA | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/gemma/README.md) |
| PyTorch | Gemma 1.1 2B | SST-2 | LoRA | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/gemma11/README.md) |
| PyTorch | Gemma 1.1 2B | Squad-V2 | LoRA | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/gemma11/README.md) |
| PyTorch | ALBERT | Banking77 | Adapters | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/albert/README.md) |
| PyTorch | Phi-1 | SST-2 | LoRA | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/phi/README.md) |
| PyTorch | Phi-1 | Squad-V2 | LoRA | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/phi/README.md) |
| PyTorch | Phi-1.5 | SST-2 | LoRA | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/phi/README.md) |
| PyTorch | Phi-1.5 | Squad-V2 | LoRA | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/phi/README.md) |
| JAX | MLP | MNIST | Full-model | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/mnist/README.md) |
| JAX | MLP | MNIST | Full-model, Data parallel | N300 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/mnist/README.md) |
| JAX | MLP | MNIST | Full-model, Tensor parallel | N300 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/mnist/README.md) |
| JAX | NeRF | Blender | Full-model | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/nerf/README.md) |
| JAX | Llama 3.2 1B | SST-2 | LoRA | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/llama/README.md) |
| JAX | Llama 3.2 1B | SST-2 | DoRA | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/llama_dora/README.md) |
| JAX | DistilBERT | SST-2 | Distillation | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/distil_bert/README.md) |
| JAX | DistilBERT | SST-2 | Distillation, Data parallel | N300 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/distil_bert/README.md) |
| Lightning | NeRF | Blender | Full-model | N150 | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/lightning/nerf/README.md) |


## Navigating the Experiment Structure
Within this repository, you'll find the following structure to help you navigate the experimental setup:

- `datasets/`: The dataset loaders for specific model training are defined in this directory and organized by the framework they utilize. For example, the loader for the MNIST dataset in PyTorch can be found at `datasets/torch/mnist/`.
- `models/`: This directory is organized by framework. Within it, you'll find subdirectories (e.g., `jax/`, `torch/`) containing the model implementations or loader scripts specific to that framework. For instance, the JAX implementation of a model for MNIST training would be located in `models/jax/mnist/`.
- `experiments/`: Experiments are organized first by the framework they utilize, and then by the specific model or task. For example, the JAX-based MNIST experiment can be found under `experiments/jax/mnist/`. Within each experiment directory, you will typically find the following files:

    - A Python file defining the configuration structure for the experiment (e.g. `configs.py`).
    - A YAML file containing the specific configuration parameters for a particular run of the experiment (e.g. `test_mnist.yaml`).
    - The Python script responsible for running the experiment using the defined configurations (e.g. `test_pure_jax_mnist.py`), which may be located within subdirectories which specify the compute environment or sharding strategy:
        - `single_chip/`: Contains experiments designed to run on a single chip.
        - `multi_chip/`: Contains experiments designed to run across multiple chips, further organized in subdirectories by sharding strategy (e.g. data-parallel or tensor-parallel). For example, full path of data-parallel MNIST training is `experiments/jax/mnist/multi_chip/data_parallel/test_pure_jax_mnist.py`.
        - If sharding strategy isn't specified, the single chip configuration is assumed.
