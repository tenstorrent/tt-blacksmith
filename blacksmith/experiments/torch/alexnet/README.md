# AlexNet Training Experiment

This directory contains the code for the AlexNet training experiment.
AlexNet model specification can be found [here](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

## Overview

The AlexNet training experiment trains an AlexNet model on the Tiny ImageNet-200 image classification dataset.
The experiment is designed to run on both CPU and GPU, with support for Tenstorrent hardware acceleration.

## Training

```bash
python3 blacksmith/experiments/torch/alexnet/test_alexnet_training.py
```

## Data

### Tiny ImageNet-200
Tiny ImageNet-200 is a subset of ImageNet with 200 classes, containing 100,000 64x64 color images.
Each class has 500 training images, 50 validation images, and 50 test images.

**Required Dataset Structure:**
```
tiny-imagenet-200/
├── wnids.txt  (class identifiers)
├── words.txt  (class descriptions)
├── train/
│   ├── n01440764/  (class folders with images)
│   ├── n01443537/
│   └── ...
├── val/
│   ├── images/  (all validation images)
│   └── val_annotations.txt  (image-to-class mapping)
└── test/
    └── images/  (test images)
```

Source: [Tiny ImageNet](https://cs231n.github.io/assignments2016/assignment3/)

Example
```
{
  "image": <256x256x3 tensor>,
  "label": 142
}
```
- image: RGB image tensor resized to 256x256
- label: Class index (0-199 for Tiny ImageNet-200)

## Configuration

The experiment is configured using the configuration file `test_alexnet_training.yaml`. The configuration file specifies the hyperparameters for the experiment, such as the number of epochs, the batch size, and the learning rate.

Current `test_alexnet_training.yaml` has the recommended and tested hyperparameters for the experiment.

### Configuration Parameters

#### Dataset Settings
| Parameter | Description | Default Value |
| --- | --- | --- |
| `dataset_id` | The dataset identifier used for training. | "tiny-imagenet" |
| `data_path` | Path to the Tiny ImageNet dataset directory. | "tiny-imagenet-200" |
| `train_ratio` | Ratio of training data (rest used for validation). | 0.9 |
| `dtype` | Data type used during training. | "torch.float32" |
| `num_workers` | Number of data loader workers. | 8 |

#### Model Settings
| Parameter | Description | Default Value |
| --- | --- | --- |
| `model_name` | AlexNet model variant. | "alexnet" |
| `input_size` | Input image size (height, width). | 256 |
| `output_size` | Number of output classes. | 200 |
| `bias` | Whether to use bias in model layers. | True |

#### Training Hyperparameters
| Parameter | Description | Default Value |
| --- | --- | --- |
| `learning_rate` | Learning rate for the optimizer. | 0.001 |
| `batch_size` | Number of samples per training batch. | 256 |
| `num_epochs` | Total number of training epochs. | 10 |
| `train_log_steps` | Frequency of training logging (in steps). | 100 |
| `val_log_epochs` | Frequency of validation logging (in epochs). | 1 |
| `loss_fn` | Loss function to use. | "torch.nn.CrossEntropyLoss" |
| `optim` | Optimizer to use for training. | "sgd" |
| `momentum` | Momentum factor for SGD optimizer. | 0.9 |
| `weight_decay` | L2 regularization factor. | 1e-4 |

#### Reproducibility Settings
| Parameter | Description | Default Value |
| --- | --- | --- |
| `seed` | Random seed for reproducibility. | 42 |
| `deterministic` | Whether to use deterministic algorithms. | False |

#### Logging Settings
| Parameter | Description | Default Value |
| --- | --- | --- |
| `log_level` | Logging level (DEBUG, INFO, WARNING, ERROR). | "INFO" |
| `use_wandb` | Whether to use Weights & Biases for logging. | True |
| `wandb_project` | Project name for Weights & Biases logging. | "alexnet-training" |
| `wandb_run_name` | Run name for Weights & Biases logging. | "alexnet-training" |
| `wandb_tags` | Tags for Weights & Biases run. | ["tt-xla", "model:torch", "plugin", "wandb"] |
| `wandb_watch_mode` | Watch mode for model parameters in wandb. | "gradients" |
| `wandb_log_freq` | Frequency of logging to wandb (in steps). | 100 |
| `model_to_wandb` | Whether to upload model to wandb. | False |
| `steps_freq` | Frequency of step-based logging. | 100 |
| `epoch_freq` | Frequency of epoch-based logging. | 1 |

#### Checkpoint Settings
| Parameter | Description | Default Value |
| --- | --- | --- |
| `resume_from_checkpoint` | Whether to resume training from a checkpoint. | False |
| `resume_option` | Checkpoint resume option ("last", "best", etc.). | "last" |
| `checkpoint_path` | Path to checkpoint file (if resuming). | "" |
| `checkpoint_metric` | Metric to use for checkpoint selection. | "val/loss" |
| `checkpoint_metric_mode` | Mode for checkpoint metric ("min" or "max"). | "min" |
| `keep_last_n` | Maximum number of last checkpoints to keep. | 3 |
| `keep_best_n` | Maximum number of best checkpoints to keep. | 1 |
| `save_strategy` | Strategy for saving checkpoints (epoch, steps, etc.). | "epoch" |
| `save_optim` | Whether to save optimizer state in checkpoints. | False |
| `project_dir` | Directory to save model checkpoints and logs. | "experiments/results/alexnet" |
| `storage_backend` | Storage backend for checkpoints ("local", etc.). | "local" |
| `sync_to_storage` | Whether to sync checkpoints to remote storage. | False |
| `load_from_storage` | Whether to load checkpoints from remote storage. | False |
| `remote_path` | Remote path for checkpoint storage. | "" |

#### Multi-chip Settings
| Parameter | Description | Default Value |
| --- | --- | --- |
| `parallelism` | Parallelism strategy ("single", "data", "tensor".). | "single" |
| `mesh_shape` | Mesh shape for multi-chip training. | "1,1" |

#### Other Settings
| Parameter | Description | Default Value |
| --- | --- | --- |
| `device` | Device to use for training ("TT", "cuda", "cpu"). | "TT" |
| `experiment_name` | Name of the experiment. | "alexnet-training" |
| `framework` | Framework used for training. | "pytorch" |
| `output_dir` | Directory for experiment outputs. | "experiments/results/alexnet" |
| `use_tt` | Whether to use Tenstorrent hardware. | True |