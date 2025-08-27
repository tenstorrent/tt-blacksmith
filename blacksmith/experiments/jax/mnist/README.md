# MNIST Experiment
## Overview
This directory contains the code in JAX (and in Flax) for training multilayer perceptron (MLP) on MNIST dataset, through Tenstorrent's Forge compiler with logging to [Weights&Biases](https://wandb.ai/site/).
The connection to the Tenstorrent device is established in script ```blacksmith/tools/jax_utils.py```. For more details on connecting to Tenstorrent's hardware in JAX, please refer to [tt-xla](https://github.com/tenstorrent/tt-xla).
## Training
To run the single chip training script in JAX, run the command
```
python3 blacksmith/experiments/jax/mnist/single_chip/test_pure_jax_mnist.py
```
To run the single chip training script in Flax, run the command
```
python3 blacksmith/experiments/jax/mnist/single_chip/test_flax_mnist.py
```
To run the multi chip training script (data or tensor parallel), run the command(s)
```
python3 blacksmith/experiments/jax/mnist/multi_chip/data_parallel/test_pure_jax_mnist.py
python3 blacksmith/experiments/jax/mnist/multi_chip/tensor_parallel/test_pure_jax_mnist.py
```
For now, data and tensor parallel strategies are supported in multi chip case while other strategies (FSDP and Pipeline parallel) are work in progress.
All commands should be run from project root directory (```tt-blacksmith```).

## Data
The [dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) consists of 60,000 training images and 10,000 test images of handwritten digits (28x28 grayscale).
The MNIST dataset is automatically downloaded in ```blacksmith/datasets/jax/mnist/dataloader.py``` when you run training scripts.
No manual download and preprocessing is required.

## Configuration
### Model architecture
- Input Layer: 784 units (flattened 28x28 MNIST images).
- Hidden Layers: Two layers with 128 units each, using ReLU activation.
- Output Layer: 10 units (one per digit).
### Training flags
- ```run_test``` runs the best model.
- ```export_shlo``` extracts [StableHLO](https://openxla.org/stablehlo) ops used in various training functions of interest (optimizer, forward pass...).
### Logging
- By default, weights, gradients and metrics are logged to [Weights&Biases](https://wandb.ai/site/). This can be turned off via ```log_on_wandb``` flag in ```logger_config.yaml```.
- Checkpointing directory is specified in ```logger_config.yaml``` as well, and user should change it's default path (```path/to/dir```) to the desired custom path. Additionally,
if ```log_on_wandb``` is ```True```, the subdirectory for that run will be automatically named after ```wandb.run.name``` - if the ```log_on_wandb``` is ```False``` user must specify
the name of the checkpointing subdirectory via ```run_name``` in ```logger_config.yaml```.
