# MNIST Linear model training experiment

This directory contains the code for the training linear model for MNIST dataset, using torch_xla environment.
Linear model is from [tt-blacksmith/blacksmith/models/torch/mnist/mnist_linear.py]

## Environment

Activate tt-xla environment, install and build tt-xla following instructions from [tt-xla](https://github.com/tenstorrent/tt-xla) and [documentation](https://docs.tenstorrent.com/tt-xla/).

## Training

Test MNIST training running [test_mnist_training.py].

```bash
python blacksmith/experiments/torch/mnist/test_mnist_training.py
```



