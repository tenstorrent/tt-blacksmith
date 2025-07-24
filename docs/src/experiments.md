# Experiments

This page provides an overview of the experiments included in this repository, detailing their organization.

## Available Experiments

The following table provides an overview of different model and dataset combinations within various frameworks explored in this project.

| Framework | Model | Dataset | Devices | Details |
| --- | --- | --- | --- | --- |
| Lightning | MLP | MNIST | TT | [README](https://github.com/tenstorrent/tt-blacksmith/tree/main/blacksmith/experiments/lightning/mnist/README.md) |
| JAX | MLP | MNIST | TT | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/mnist/README.md) |
| JAX | NeRF | Blender | TT | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/jax/nerf/README.md) |
| Lightning | NeRF | Blender | TT | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/lightning/nerf/README.md) |
| PyTorch | Llama | SST-2 | GPU | [README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/README.md) |

## Navigating the Experiment Structure
Within this repository, you'll find the following structure to help you navigate the experimental setup:

- `datasets/`: The dataset loaders for specific model training are defined in this directory and organized by the framework they utilize. For example, the loader for the MNIST dataset can be found at `datasets/mnist/`.
- `models/`: This directory is organized by framework. Within it, you'll find subdirectories (e.g., `jax/`, `pytorch/`) containing the model implementations or loader scripts specific to that framework. For instance, the JAX implementation of a model for MNIST training would typically be located in `models/jax/mnist/`.
- `experiments/`: Experiments are organized first by the framework they utilize, and then by the specific model or task. For example, the JAX-based MNIST experiment can be found under `blacksmith/experiments/jax/mnist/`. Within each experiment directory, you will typically find the following files:

    - A Python file defining the configuration structure for the experiment (e.g. `configs.py`).
    - A YAML file containing the specific configuration parameters for a particular run of the experiment (e.g. `test_jax_mnist.yaml`).
    - The Python script responsible for running the experiment using the defined configurations (e.g. `test_pure_jax_mnist.py`), which may be located within subdirectories which specify the compute environment or sharding strategy:
        - `single_chip/`: Contains experiments designed to run on a single chip.
        - `multi_chip/`: Contains experiments designed to run across multiple chips, further organized in subdirectories by sharding strategy (e.g. data-parallel or tensor-parallel). For example, full path of data-parallel MNIST training is `blacksmith/experiments/jax/mnist/multi_chip/data_parallel/test_pure_jax_mnist.py`.
        - If sharding strategy isn't specified, the single chip configuration is assumed for now.
