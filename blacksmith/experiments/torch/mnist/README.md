# MNIST Model Training Experiments

This directory contains the code for training models on the MNIST dataset using the torch_xla environment. Two model architectures are supported:

1. **Linear model** (`MNISTLinear`) - A simple fully-connected network from `tt-blacksmith/blacksmith/models/torch/mnist/mnist_linear.py`
2. **Convolutional model** (`MNISTCNN`) - A CNN with convolutional layers from `tt-blacksmith/blacksmith/models/torch/mnist/mnist_cnn.py`

## Setup

Activate `tt-xla` environment using the provided scripts from `tt-blacksmith` documentation.

## Training

### 1. Single Chip - Linear Model

Test MNIST training with linear model running [test_mnist_training.py].

```bash
python blacksmith/experiments/torch/mnist/test_mnist_training.py
```

### 2. Single Chip - CNN Model

Test MNIST training with convolutional model running [test_mnist_cnn_training.py].

```bash
python blacksmith/experiments/torch/mnist/cnn/test_mnist_cnn_training.py
```

### 3. Multichip - data parallel

Test MNIST DP training with linear model running [test_mnist_training.py].

```bash
python blacksmith/experiments/torch/mnist/data_parallel/test_mnist_training.py
```

### 4. Multichip - tensor parallel

Test MNIST TP training with linear model running [test_mnist_training.py].

```bash
python blacksmith/experiments/torch/mnist/tensor_parallel/test_mnist_training.py
```


## Configuration

For each training you can change default values in configuration files:
1. Single chip - Linear - `blacksmith/experiments/torch/mnist/test_mnist_training.yaml`
2. Single chip - CNN - `blacksmith/experiments/torch/mnist/cnn/test_mnist_cnn_training.yaml`
3. Data parallel - `blacksmith/experiments/torch/mnist/data_parallel/test_mnist_training_dp.yaml`
4. Tensor parallel - `blacksmith/experiments/torch/mnist/tesnor_parallel/test_mnist_training_tp.yaml`

### Linear Model Configuration

In `blacksmith/experiments/torch/mnist/test_mnist_training.yaml` you can change default values for following parameters.

| Parameter | Description | Default Value |
| --- | --- | --- |
|  **Dataset Settings** |
| `dataset_id` | Name of the dataset. | "mnist" |
| `train_ratio` | Training/Validation dataset ratio. | 0.8 |
| `dtype` | Data type used for input tensors. | "torch.bfloat16" |
|  **Model Settings** |
| `model_name` | Name of the model architecture. | "MNISTLinear" |
| `input_size` | Number of input features (e.g., flattened image size). | 784 |
| `hidden_size` | Size of the hidden layer in the model. | 512 |
| `output_size` | Number of output classes. | 10 |
| `bias` | Whether to include bias terms in the layers. | false |
|  **Training Hyperparameters** |
| `learning_rate` | Learning rate used by the optimizer. | 0.01 |
| `batch_size` | Number of samples per training batch. | 256 |
| `num_epochs` | Total number of training epochs. | 16 |
| `train_log_steps` | Number of training steps between logging. | 100 |
| `val_log_epochs` | Number of epochs between validation logging. | 5 |
| `loss_fn` | The loss function used for training. | "torch.nn.MSELoss" |
| `optim` | Optimizer to use. | "sgd" |
|  **Reproducibility Settings** |
| `seed` | Random seed for reproducibility. | 23 |
| `deterministic` | Whether to use deterministic algorithms. | false |
|  **Logging Settings** |
| `log_level` | Logging level. | "INFO" |
| `use_wandb` | Whether to use Weights & Biases for logging. | true |
| `wandb_project` | W&B project name. | "blacksmith-mnist" |
| `wandb_run_name` | W&B run name. | "mnist_single_chip" |
| `wandb_tags` | A list of tags for the experiment. | ["tt-xla", "model:torch", "plugin", "wandb"] |
| `wandb_watch_mode` | W&B watch mode for model tracking. | "all" |
| `wandb_log_freq` | Frequency of W&B logging. | 100 |
| `model_to_wandb` | Whether to log model to W&B. | false |
| `steps_freq` | Frequency of step logging. | 100 |
| `epoch_freq` | Frequency of epoch logging. | 5 |
|  **Checkpoint Settings** |
| `resume_from_checkpoint` | Whether to resume from a checkpoint. | false |
| `resume_option` | Checkpoint resume option: "last", "best", or "path". | "last" |
| `checkpoint_path` | Path to checkpoint if resume_option is "path". | "" |
| `checkpoint_metric` | Metric to use for checkpoint selection. | "val/loss" |
| `checkpoint_metric_mode` | Whether to minimize or maximize checkpoint metric: "min" or "max". | "min" |
| `keep_last_n` | Number of last checkpoints to keep. | 3 |
| `keep_best_n` | Number of best checkpoints to keep. | 1 |
| `save_strategy` | Checkpoint save strategy. | "epoch" |
| `project_dir` | Project directory path. | "blacksmith/experiments/torch/mnist" |
| `save_optim` | Whether to save optimizer state. | false |
| `storage_backend` | Storage backend for checkpoints. | "local" |
| `sync_to_storage` | Whether to sync checkpoints to storage. | false |
| `load_from_storage` | Whether to load checkpoints from storage. | false |
| `remote_path` | Remote path for checkpoint storage. | "" |
|  **Multi-chip settings** |
| `parallelism` | Select experiment - "single"/"data"/"tensor". | "" |
| `mesh_shape` | Mesh shape. | "2,1" |
|  **Other Settings** |
| `device` | Select device "TT"/"CPU". | "TT" |
| `experiment_name` | The name of the experiment used for tracking and logging. | "torch-mnist" |
| `framework` | Framework being used. | "pytorch" |
| `output_dir` | Output directory for results. | "experiments/results/mnist" |
| `use_tt` | Whether to use TT device. | true |

### CNN Model Configuration

In `blacksmith/experiments/torch/mnist/cnn/test_mnist_cnn_training.yaml` you can change values for following parameters.

Most parameters are the same as the Linear model configuration above, with the following CNN-specific model settings:

| Parameter | Description | Default Value |
| --- | --- | --- |
|  **Model Settings (CNN-specific)** |
| `model_name` | Name of the model architecture. | "MNISTCNN" |
| `conv1_channels` | Output channels from first convolutional layer. | 32 |
| `conv2_channels` | Output channels from second convolutional layer. | 64 |
| `fc1_size` | Size of first fully connected layer. | 128 |
| `output_size` | Number of output classes. | 10 |
| `dropout1_rate` | Dropout rate after max pooling. | 0.25 |
| `dropout2_rate` | Dropout rate before final layer. | 0.5 |
| `bias` | Whether to include bias terms in the layers. | false |
|  **Other Settings** |
| `experiment_name` | The name of the experiment used for tracking and logging. | "torch-mnist-cnn" |
| `output_dir` | Output directory for results. | "experiments/results/mnist_cnn" |
| `wandb_project` | W&B project name. | "blacksmith-mnist-cnn" |
| `wandb_run_name` | W&B run name. | "mnist_cnn_single_chip" |
