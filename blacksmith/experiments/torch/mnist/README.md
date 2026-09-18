# MNIST Model Training Experiments

This directory contains the code for training the linear MLP model on the MNIST dataset using the torch_xla environment.

- **Linear model** (`MNISTLinear`) - A simple fully-connected network from `tt-blacksmith/blacksmith/models/torch/mnist/mnist_linear.py`

For CNN-based MNIST training, see [`blacksmith/experiments/torch/mnist_cnn/`](../mnist_cnn/README.md).

## Setup

Activate `tt-xla` environment using the provided scripts from `tt-blacksmith` documentation.

## Training

### 1. Single Chip - Linear Model

Test MNIST training with linear model running [train.py].

```bash
python blacksmith/experiments/torch/mnist/train.py
```

### 2. Multichip - data parallel

Test MNIST DP training with linear model running [train.py].

```bash
python blacksmith/experiments/torch/mnist/data_parallel/train.py
```

### 3. Multichip - tensor parallel

Test MNIST TP training with linear model running [train.py].

```bash
python blacksmith/experiments/torch/mnist/tensor_parallel/train.py
```


## Configuration

| Architecture       | mesh_shape                   | mesh_axis_names      | input_sharding_dim | dataset      | Method                      |
| ------------------ | ---------------------------- | -------------------- | ------------------ | ------------ | --------------------------- |
| [Single-Chip](single_chip/mnist.yaml) | None      | None                 | None               | MNIST        | Full Model                  |
| [N300](data_parallel/mnist_dp.yaml)   | `[1, 2]`  | `["model", "batch"]` | `"batch"`          | MNIST        | Full Model, Data Parallel   |
| [N300](tensor_parallel/mnist_tp.yaml) | `[1, 2]`  | `["batch", "model"]` | None               | MNIST        | Full Model, Tensor Parallel |

For each training you can change default values in configuration files:
1. Single chip - Linear - `blacksmith/experiments/torch/mnist/single_chip/mnist.yaml`
2. Data parallel - `blacksmith/experiments/torch/mnist/data_parallel/mnist_dp.yaml`
3. Tensor parallel - `blacksmith/experiments/torch/mnist/tensor_parallel/mnist_tp.yaml`

### Linear Model Configuration

In `blacksmith/experiments/torch/mnist/single_chip/mnist.yaml` you can change default values for following parameters.

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
| `project_dir` | Project directory path. | "blacksmith/experiments/torch/mnist" |
| `save_optim` | Whether to save optimizer state. | false |
| `storage_backend` | Storage backend for checkpoints. | "local" |
| `sync_to_storage` | Whether to sync checkpoints to storage. | false |
| `load_from_storage` | Whether to load checkpoints from storage. | false |
| `remote_path` | Remote path for checkpoint storage. | "" |
|  **Multi-chip settings** |
| `mesh_shape` | Mesh shape. | None |
| `mesh_axis_names` | Axis names for the mesh. | None |
|  **Other Settings** |
| `device` | Select device "TT"/"CPU". | "TT" |
| `experiment_name` | The name of the experiment used for tracking and logging. | "torch-mnist" |
| `framework` | Framework being used. | "pytorch" |
| `output_dir` | Output directory for results. | "experiments/results/mnist" |
| `use_tt` | Whether to use TT device. | true |
