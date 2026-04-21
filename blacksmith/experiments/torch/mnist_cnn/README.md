# MNIST CNN Training Experiment

This directory contains the code for training a convolutional model on the MNIST dataset using the torch_xla environment.

- **Convolutional model** (`MNISTCNN`) - A CNN with convolutional layers from `tt-blacksmith/blacksmith/models/torch/mnist/mnist_cnn.py`

## Setup

Activate `tt-xla` environment using the provided scripts from `tt-blacksmith` documentation.

## Training

### Single Chip - CNN Model

```bash
python blacksmith/experiments/torch/mnist_cnn/train.py
```

## Configuration

In `blacksmith/experiments/torch/mnist_cnn/single_chip/mnist_cnn.yaml` you can change values for following parameters.

| Parameter | Description | Default Value |
| --- | --- | --- |
|  **Dataset Settings** |
| `dataset_id` | Name of the dataset. | "mnist" |
| `train_ratio` | Training/Validation dataset ratio. | 0.8 |
| `dtype` | Data type used for input tensors. | "torch.bfloat16" |
|  **Model Settings** |
| `model_name` | Name of the model architecture. | "MNISTCNN" |
| `conv1_channels` | Output channels from first convolutional layer. | 32 |
| `conv2_channels` | Output channels from second convolutional layer. | 64 |
| `kernel_size` | Convolutional kernel size. | 3 |
| `stride` | Convolutional stride. | 1 |
| `fc1_size` | Size of first fully connected layer. | 128 |
| `output_size` | Number of output classes. | 10 |
| `dropout1_rate` | Dropout rate after max pooling. | 0.25 |
| `dropout2_rate` | Dropout rate before final layer. | 0.5 |
| `bias` | Whether to include bias terms in the layers. | false |
|  **Training Hyperparameters** |
| `learning_rate` | Learning rate used by the optimizer. | 0.01 |
| `batch_size` | Number of samples per training batch. | 256 |
| `num_epochs` | Total number of training epochs. | 16 |
| `loss_fn` | The loss function used for training. | "torch.nn.CrossEntropyLoss" |
| `optim` | Optimizer to use. | "sgd" |
|  **Reproducibility Settings** |
| `seed` | Random seed for reproducibility. | 23 |
| `deterministic` | Whether to use deterministic algorithms. | false |
|  **Logging Settings** |
| `log_level` | Logging level. | "INFO" |
| `use_wandb` | Whether to use Weights & Biases for logging. | true |
| `wandb_project` | W&B project name. | "blacksmith-mnist-cnn" |
| `wandb_run_name` | W&B run name. | "mnist_cnn_single_chip" |
| `wandb_tags` | A list of tags for the experiment. | ["tt-xla", "model:torch", "cnn", "plugin", "wandb"] |
| `wandb_watch_mode` | W&B watch mode for model tracking. | "all" |
| `wandb_log_freq` | Frequency of W&B logging. | 100 |
| `model_to_wandb` | Whether to log model to W&B. | false |
| `steps_freq` | Frequency of step logging. | 100 |
| `val_steps_freq` | Frequency of step validations. | 100 |
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
| `project_dir` | Project directory path. | "blacksmith/experiments/torch/mnist_cnn" |
| `save_optim` | Whether to save optimizer state. | false |
| `storage_backend` | Storage backend for checkpoints. | "local" |
| `sync_to_storage` | Whether to sync checkpoints to storage. | false |
| `load_from_storage` | Whether to load checkpoints from storage. | false |
| `remote_path` | Remote path for checkpoint storage. | "" |
|  **Device Settings** |
| `mesh_shape` | Mesh shape. | None |
| `mesh_axis_names` | Axis names for the mesh. | None |
|  **Other Settings** |
| `experiment_name` | The name of the experiment used for tracking and logging. | "torch-mnist-cnn" |
| `framework` | Framework being used. | "pytorch" |
| `output_dir` | Output directory for results. | "experiments/results/mnist_cnn" |
| `use_tt` | Whether to use TT device. | true |
