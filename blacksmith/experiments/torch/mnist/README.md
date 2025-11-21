# MNIST Linear model training experiment

This directory contains the code for the training linear model for MNIST dataset, using torch_xla environment.
Linear model is from *tt-blacksmith/blacksmith/models/torch/mnist/mnist_linear.py*

## Setup

Activate `tt-xla` environment using the provided scripts from `tt-blacksmith` documentation.

## Training

### 1. Single chip

Test MNIST training running [test_mnist_training.py].

```bash
python blacksmith/experiments/torch/mnist/test_mnist_training.py
```

### 2. Multichip - data parallel

Test MNIST DP training running [test_mnist_training.py].

```bash
python blacksmith/experiments/torch/mnist/data_parallel/test_mnist_training.py
```

### 3. Multichip - tensor parallel

Test MNIST TP training running [test_mnist_training.py].

```bash
python blacksmith/experiments/torch/mnist/tensor_parallel/test_mnist_training.py
```


## Configuration

For each training you can change default values in configuration files:
1. Single chip - `blacksmith/experiments/torch/mnist/test_mnist_training.yaml`
2. Data parallel - `blacksmith/experiments/torch/mnist/data_parallel/test_mnist_training_dp.yaml`
3. Tensor parallel - `blacksmith/experiments/torch/mnist/tesnor_parallel/test_mnist_training_tp.yaml`

In `blacksmith/experiments/torch/mnist/test_mnist_training.yaml` you can for example change default values for following parameters.

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
