# MNIST Linear model training experiment

This directory contains the code for the training linear model for MNIST dataset, using torch_xla environment.
Linear model is from *tt-blacksmith/blacksmith/models/torch/mnist/mnist_linear.py*

## Setup

Activate `tt-xla` environment using the provided scripts from `tt-blacksmith` documentation.

## Training

Test MNIST training running [test_mnist_training.py].

```bash
python blacksmith/experiments/torch/mnist/test_mnist_training.py
```

## Configuration

In `blacksmith/experiments/torch/mnist/test_mnist_training.yaml` you can for example change default values for following parameters.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `experiment_name` | The name of the experiment used for tracking and logging. | "blacksmith-mnist-training" |
| `tags` | A list of tags for the experiment. | ["tt-xla", "model:torch", "plugin", "wandb"] |
|  **Device** |
| `device` | Select device "TT"/"CPU". | "TT" |
|  **Training** |
| `training_config.train_ratio` | Training/Validation dataset ratio. | 0.8 |
| `training_config.batch_size` | Number of samples per training batch. | 64 |
| `training_config.epochs` | Total number of training epochs. | 5 |
| `training_config.lr` | Learning rate used by the optimizer. | 0.001 |
|  **Model (Net Config)** |
| `net_config.input_size` | Number of input features (e.g., flattened image size). | 784 |
| `net_config.hidden_size` | Size of the hidden layer in the model. | 512 |
| `net_config.output_size` | Number of output classes. | 10 |
| `net_config.bias` | Whether to include bias terms in the layers. | True |
|  **Data Loading** |
| `data_loading_config.batch_size` | Batch size used during data loading. | 64 |
| `data_loading_config.dtype` | Data type used for input tensors. | torch.bfloat16 |
|  **Loss Function** |
| `loss` | The loss function used for training. | torch.nn.MSELoss |
