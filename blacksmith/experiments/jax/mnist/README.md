# MNIST Experiment
## Overview
This directory contains the code in JAX (and in Flax) for training multilayer perceptron (MLP) on MNIST dataset, through Tenstorrent's Forge compiler with logging to [Weights&Biases](https://wandb.ai/site/).
The connection to the Tenstorrent device is established in script ```blacksmith/tools/jax_utils.py```. For more details on connecting to Tenstorrent's hardware in JAX, please refer to [tt-xla](https://github.com/tenstorrent/tt-xla).
## Training
To run the training script in JAX, run the command
```
python3 blacksmith/experiments/jax/mnist/test_pure_jax_mnist.py
```
To run the training script in Flax, run the command
```
python3 blacksmith/experiments/jax/mnist/test_flax_mnist.py
```
Both should be run from project root directory (```tt-blacksmith```).

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
