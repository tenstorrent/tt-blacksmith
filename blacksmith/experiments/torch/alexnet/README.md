# AlexNet Training Experiment

This directory contains the code for the AlexNet training experiment.
AlexNet model specification can be found [here](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).

## Overview

The AlexNet training experiment trains an AlexNet model on the Tiny ImageNet-200 image classification dataset.
The experiment is designed to run on both CPU and GPU, with support for Tenstorrent hardware acceleration.

NOTE: AlexNet training currently supports both CPU and GPU execution. Tenstorrent hardware acceleration is being developed.

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
  "image": <64x64x3 tensor>,
  "label": 142
}
```
- image: RGB image tensor resized to 64x64
- label: Class index (0-199 for Tiny ImageNet-200)

## Configuration

The experiment is configured using the configuration file `test_alexnet_training.yaml`. The configuration file specifies the hyperparameters for the experiment, such as the number of epochs, the batch size, and the learning rate.

Current `test_alexnet_training.yaml` has the recommended and tested hyperparameters for the experiment.

### Configuration Parameters

| Parameter | Description | Default Value|
| --- | --- | --- |
| `dataset_name` | The dataset used for training. | "tiny-imagenet" |
| `data_path` | Path to the Tiny ImageNet dataset directory. | "tiny-imagenet-200" |
| `model_variant` | AlexNet model variant. | "alexnet" |
| `num_classes` | Number of output classes. | 200 |
| `image_size` | Input image size (height, width). | 64 |
| `dtype` | Data type used during training. | "torch.float32" |
| `learning_rate` | Learning rate for the optimizer. | 0.01 |
| `batch_size` | Number of samples per training batch. | 256 |
| `momentum` | Momentum factor for SGD optimizer. | 0.9 |
| `weight_decay` | L2 regularization factor. | 1e-4 |
| `num_epochs` | Total number of training epochs. | 90 |
| `optim` | Optimizer to use for training. | "sgd" |
| `lr_scheduler` | Learning rate scheduler type. | "step" |
| `lr_step_size` | Step size for StepLR scheduler. | 30 |
| `lr_gamma` | Gamma factor for StepLR scheduler. | 0.1 |
| `seed` | Random seed for reproducibility. | 42 |
| `output_dir` | Directory to save model checkpoints and logs. | "experiments/results/alexnet" |
| `report_to` | Backend for experiment tracking. | "wandb" |
| `wandb_project` | Project name for Weights & Biases logging. | "alexnet-training" |
| `wandb_watch_mode` | Watch mode for model parameters in wandb. | "gradients" |
| `wandb_log_freq` | Frequency of logging to wandb (in steps). | 100 |
| `save_strategy` | Strategy for saving checkpoints (epoch, steps, etc.). | "epoch" |
| `logging_strategy` | Strategy for logging (steps, epoch, etc.). | "steps" |
| `logging_steps` | Frequency of logging (in steps). | 100 |
| `save_total_limit` | Maximum number of checkpoints to keep. | 3 |
| `do_train` | Whether to run training. | True |
| `do_eval` | Whether to run evaluation. | True |
| `eval_split` | Evaluation split ratio. | 0.1 |
| `num_workers` | Number of data loader workers. | 8 | 