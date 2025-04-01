# NeRF Experiment

This directory contains the code for the NeRF experiment.
Original paper can be found [here](https://arxiv.org/pdf/2206.00878).

## Overview

The NeRF experiment is an implementation of the EfficientNeRF algorithm for 3D reconstruction from 2D images.
The experiment is designed to run on the lightning framework.

## Training

```bash
python3 thomas/experiments/lightning/nerf/test_nerf.py
```

## Data

The experiment uses the Blender dataset, which is a synthetic dataset of 3D objects.
It can be found [here](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset).

The dataset should follow the following structure:

```
data/
├── train/
│   ├── r_0.png
│   ├── r_1.png
│   └── ...
├── test/
│   ├── r_0.png
│   ├── r_1.png
│   └── ...
├── transforms_train.json
└── transforms_test.json
```

The `transforms_train.json` and `transforms_test.json` files contain the camera parameters for the images in the dataset.
Check the dataset documentation mentioned above for more information.


## Configuration

The experiment is configured using the configuration file `test_nerf.yaml`. The configuration file specifies the hyperparameters for the experiment, such as the number of epochs, the batch size, and the learning rate.

Current `test_nerf.yaml` has the recommended and tested hyperparameters for the experiment.

### Configuration Paramaters

| Parameter | Description | Default Value |
| --- | --- | --- |
| `experiment_name` | The name of the experiment used for wandb. | "nerf-training" |
| `tags` | List of tags for the exepriment used for wandb | ["nerf"] |
|  **Data Loading** |
| `data_loading.input_dir` | Input directory for dataset. | "./data/nerf_synthetic/lego" |
| `data_loading.batch_size` | The batch size (# rays) for the data loader. | 1024 |
| `data_loading.img_wh` | Image width and height | 800 |
|  **Training** |
| `training.use_forge` | Run model through compiler or not | False |
| `training.device` | Choose the device to run on. Can be "cpu", "cuda", etc. | cpu |
| `training.val_only` | To run training or validation | False |
| `training.epochs` | The number of epochs to train the model for. | 16 |
| `training.loss` | The loss function to use. | "mse" |
| `training.optimizer` | The optimizer to use for training. | "radam" |
| `training.optimizer_kwargs` | Optimizer kwargs. | None |
| `training.lr_scheduler` | The scheduler to use for the learning rate. | "cosine" |
| `training.lr_scheduler_kwargs` | Kwargs for the learning rate scheduler. | None |
| `training.warmup_multiplier` | The multiplier for the warmup steps. | 1.0 |
| `training.warmup_epochs` | The number of warmup epochs. | 0 |
| `training.ckpt_path` | The path to the checkpoint file. | None |
| `training.log_every` | The number of steps to log the training loss. | 5 |
| `training.log_dir` | The directory to save the logs. | "./logs" |
|  **Model** |
| `model.deg` | The degree of the spherical harmonics. | 5 |
| `model.num_freqs` | The number of frequency bands to use for embedding. | 10 |
| `model.coarse.depth` | The depth of the coarse NeRF model. | 4 |
| `model.coarse.width` | The width of the coarse NeRF model. | 8 |
| `model.coarse.samples` | The number of samples to use for the coarse NeRF model. | 8 (recommended 64) |
| `model.fine.depth` | The depth of the fine NeRF model. | 4 |
| `model.fine.width` | The width of the fine NeRF model. | 8 |
| `model.fine.samples` | The number of samples to use for the fine NeRF model. | 8 |
| `model.coord_scope` | The scope of the coordinates. | 3.0 |
| `model.sigma_init` | The initial value for sigma. | 30.0 |
| `model.sigma_default` | The default value for sigma. | -20.0 |
| `model.weight_threshold` | The threshold for the weights. | 0.0001 |
| `model.uniform_ratio` | The ratio of uniform samples. | 0.01 |
| `model.beta` | Beta value used for nerftree. Controls updating rate of voxels. | 0.1 |
| `model.warmup_step` | Warmup steps used for nerftree | 0 |
| `model.in_channels_dir` | Number of channels for direction tensor | 32 |
| `model.in_channels_xyz` | Number of channels for position tensor | 63 |
