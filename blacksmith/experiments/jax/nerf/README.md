# NeRF Experiment

This directory contains the code for the NeRF experiment.
Original paper can be found [here](https://arxiv.org/pdf/2206.00878).
Experiment from this directory has a minor difference from original paper; instead of using a threshold to determine important voxels, "top k" method is used.
This ensures robustness across different datasets and is more compatible with JAX framework.

## Overview

The NeRF experiment is an implementation of the EfficientNeRF algorithm for 3D reconstruction from 2D images.
The experiment is designed to run in the JAX framework.

## Training

```bash
python3 blacksmith/experiments/jax/nerf/test_nerf.py
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
| `project_name` | The name of the wandb project where experiments are logged. | "jax-nerf" |
|  **Data Loading** |
| `data_loading.dataset_name` | Hugging Face dataset name. | "Tenstorrent/tt-nerf-p150-white" |
| `data_loading.batch_size` | The batch size (# rays) for the data loader. | 4096 |
| `data_loading.img_wh` | Image width and height | 400 |
|  **Training** |
| `training.epochs` | The number of epochs to train the model for. | 15 |
| `training.loss` | The loss function to use. | "mse" |
| `training.optimizer` | The optimizer to use for training. | "radam" |
| `training.lr` | Learning rate for training. | 0.0008 |
| `training.betas` | Beta parameters used in adam or radam optimizer. | [0.9, 0.999] |
| `training.eps` | Epsilon parameter for numeric stability used in adam or radam optimizer. | 1e-8 |
| `training.log_every` | The number of steps to perform validation. | 200 |
| `training.log_on_wandb` | Flag indicating whether to use wandb or not.  | True |
| `training.cache_voxels_fine` | Flag indicating whether to cache the fine voxels or not.  | False |
| `training.resume` | Flag indicating whether to resume training from the latest checkpoint (if existent) or not.  | False |
| `training.render` | Flag indicating whether to render the images or not.  | False |
|  **Model** |
| `model.deg` | The degree of the spherical harmonics. | 2 |
| `model.num_freqs` | The number of frequency bands to use for embedding. | 10 |
| `model.coarse.depth` | The depth of the coarse NeRF model. | 4 |
| `model.coarse.width` | The width of the coarse NeRF model. | 128 |
| `model.coarse.samples` | The number of samples to use for the coarse NeRF model. | 64 |
| `model.fine.depth` | The depth of the fine NeRF model. | 4 |
| `model.fine.width` | The width of the fine NeRF model. | 192 |
| `model.fine.samples` | The number of samples to use for the fine NeRF model. | 8 |
| `model.coord_scope` | The scope of the coordinates. | 3.0 |
| `model.sigma_init` | The initial value for sigma. | 30.0 |
| `model.sigma_default` | The default value for sigma. | -20.0 |
| `model.uniform_ratio` | The ratio of uniform samples. | 0.01 |
| `model.beta` | Beta value used for nerftree. Controls updating rate of voxels. | 0.1 |
| `model.warmup_step` | Warmup steps used for nerftree | 0 |
| `model.in_channels_dir` | Number of channels for direction tensor | 32 |
| `model.in_channels_xyz` | Number of channels for position tensor | 63 |
|  **Checkpoint** |
| `checkpoint.save_dir` | Name of directory where checkpoints are saved. | path/to/dir (change to desired custom path) |
| `checkpoint.render_dir` | Name of directory where rendered images are saved. | path/to/dir (change to desired custom path) |
| `checkpoint.save_every` | Number of training steps after which checkpoints are saved. | 200 |
| `checkpoint.keep_last` | Number of most recent checkpoints that are kept. | 3 |
