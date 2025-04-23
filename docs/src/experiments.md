# Experiments

This page provides an overview of the experiments included in this repository, detailing their organization.

## Navigating the Experiment Structure
Within this repository, you'll find the following structure to help you navigate the experimental setup:

- `datasets/`: The dataset loaders are for specific model training are defined in this directory. For example, the loader for the MNIST dataset can be found at `datasets/mnist/`.
- `models/`: This directory is organized by framework. Within it, you'll find subdirectories (e.g., `jax/`, `pytorch/`) containing the model implementations or loader scripts specific to that framework. For instance, the JAX implementation of a model for MNIST training would typically be located in `models/jax/mnist/`.
- `experiments/`: Experiments are organized first by the framework they utilizes, and then by the specific model or task. For example, the JAX-based MNIST experiment can be found under `blacksmith/experiments/jax/mnist/`. Within each experiment directory, you will typically find the following files:

- A Python file defining the configuration structure for the experiment (e.g. `configs.py`).
- A YAML file containing the specific configuration parameters for a particular run of the experiment (e.g. `test_jax_mnist.yml`).
- The Python script responsible for running the experiment using the defined configurations (e.g. `test_pure_jax_mnist.py`).

## Available experiments

The following table provides an overview of different model and dataset combinations within various frameworks explored in this project.

| Framework | Model | Dataset | Details |
| --- | --- | --- |
| Lightning | MLP | MNIST | [readme](blacksmith/experiments/lightning/mnist/test_mnist_lightning_ffe.py) |
| JAX | MLP | MNIST | [readme](blacksmith/experiments/jax/mnist/test_pure_jax_mnist.py) |
| Lightning | NeRF | Blender | [readme](blacksmith/experiments/lightning/nerf/README.md) |
| PyTorch | Llama | SST-2 | [readme](blacksmith/experiments/pytorch/llama/README.md) |