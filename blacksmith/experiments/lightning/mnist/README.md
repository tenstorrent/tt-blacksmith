# MNIST Clasification experiment

This directory contains the code for the NeRF experiment.

## Overview

The experiments implements code for lightning framweork that does training ot Tenstorrent hardware or on cpu.
Currently it runs model forward and backward as well as loss forward and backward on the tt hardware.

## Training

```bash
python3 blacksmith/experiments/lightning/mnist/test_mnist_lightning_ffe.py
```

## Data

The [dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) consists of 60,000 training images and 10,000 test images of handwritten digits (28x28 grayscale).

## Configuration

In `blacksmith/experiments/lightning/mnist/test_mnist_lightning_ffe.yaml` you can for example change loss to be on cpu or pick some other loss that forge-fe have to offer.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `experiment_name` | The name of the experiment used for tracking and logging. | "blacksmith-mnist" |
| `tags` | A list of tags for the experiment (e.g., frameworks, model type). | ["tt-forge-fe", "model:torch", "lightning"] |
| `loss` | The loss function used for training. | forge.op.loss.CrossEntropyLoss |
| `dataset_id` | The training dataset id. | mnist |
| `dtype` | Data type used for input tensors. | torch.float32 |
|  **Training** |
| `training_config.batch_size` | Number of samples per training batch. | 64 |
| `training_config.epochs` | Total number of training epochs. | 2 |
| `training_config.lr` | Learning rate used by the optimizer. | 0.001 |
|  **Model (Net Config)** |
| `net_config.input_size` | Number of input features (e.g., flattened image size). | 784 |
| `net_config.hidden_size` | Size of the hidden layer in the model. | 512 |
| `net_config.output_size` | Number of output classes. | 10 |
| `net_config.bias` | Whether to include bias terms in the layers. | True |
