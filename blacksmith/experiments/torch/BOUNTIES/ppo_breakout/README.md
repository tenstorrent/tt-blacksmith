# PPO Breakout Training

This directory contains the code for training a PPO (Proximal Policy Optimization) agent on the Atari Breakout game.
The agent uses a CNN architecture from `blacksmith/models/torch/BOUNTIES/ppo_breakout/model.py` with standard Atari preprocessing wrappers.


## Overview

The experiment trains a PPO agent on `ALE/Breakout-v5` using standard Atari preprocessing (frame skipping, grayscale, 84x84 resize, frame stacking) and environment wrappers (episodic life, fire reset, reward clipping). GAE (Generalized Advantage Estimation) is used for advantage computation.

> **Note:** This experiment currently supports CPU only. TT-XLA support is not yet available.

## Training

```bash
python blacksmith/experiments/torch/BOUNTIES/ppo_breakout/test_breakout_ppo_training.py
```

## Configuration

The experiment is configured using the configuration file `test_breakout_ppo_training.yaml`. Current defaults are the recommended and tested hyperparameters.

### Configuration Parameters

| Parameter | Description | Default Value |
| --- | --- | --- |
| **Environment Settings** |
| `num_envs` | Number of parallel environments. | 8 |
| `frame_stack` | Number of consecutive frames stacked as observation. | 4 |
| `frame_skip` | Number of frames each action is repeated for. | 4 |
| **Training Hyperparameters** |
| `total_timesteps` | Total environment steps to train for. | 10_000_000 |
| `learning_rate` | Learning rate for Adam optimizer. | 2.5e-4 |
| `anneal_lr` | Whether to linearly anneal the learning rate to zero. | True |
| **PPO Hyperparameters** |
| `num_steps` | Number of rollout steps per environment per update. | 128 |
| `gamma` | Discount factor. | 0.99 |
| `gae_lambda` | Lambda for GAE advantage estimation. | 0.95 |
| `num_minibatches` | Number of minibatches per update. | 4 |
| `update_epochs` | Number of epochs per PPO update. | 4 |
| `clip_coef` | PPO clipping coefficient. | 0.1 |
| `norm_adv` | Whether to normalize advantages. | True |
| `clip_vloss` | Whether to clip value loss. | True |
| `ent_coef` | Entropy bonus coefficient. | 0.01 |
| `vf_coef` | Value function loss coefficient. | 0.5 |
| `max_grad_norm` | Maximum gradient norm for clipping. | 0.5 |
| **Logging Settings** |
| `log_level` | Logging verbosity level. | "INFO" |
| `log_interval` | Frequency of metric logging (in updates). | 1 |
| `use_wandb` | Whether to enable Weights & Biases logging. | True |
| `wandb_project` | Project name for Weights & Biases logging. | "ALE-Breakout-PPO" |
| `wandb_run_name` | Run name for Weights & Biases tracking. | "tt-breakout-ppo" |
| `wandb_tags` | List of tags assigned to the W&B run. | ["ppo", "atari", "breakout"] |
| `wandb_watch_mode` | Watch mode for model parameter logging. | "all" |
| `wandb_log_freq` | Frequency of logging to Weights & Biases (in steps). | 1000 |
| `model_to_wandb` | Whether to store model checkpoint in Weights & Biases. | False |
| **Checkpoint Settings** |
| `resume_from_checkpoint` | Whether to resume training from a previous checkpoint. | False |
| `resume_option` | Resume method (`last`, `best`, or `path`). | "last" |
| `checkpoint_path` | Path to a checkpoint if `resume_option="path"`. | "" |
| `checkpoint_metric` | Metric to monitor for best checkpoint. | "charts/avg_return" |
| `checkpoint_metric_mode` | Mode for checkpoint metric (`min` or `max`). | "max" |
| `keep_last_n` | Number of recent checkpoints to keep. | 3 |
| `keep_best_n` | Number of best checkpoints to keep. | 3 |
| `save_strategy` | Strategy for saving checkpoints. | "step" |
| `save_interval` | Frequency of checkpoint saving (in updates). | 50 |
| `project_dir` | Directory for experiment outputs. | "blacksmith/experiments/torch/BOUNTIES/ppo_breakout" |
| `save_optim` | Whether to save optimizer state. | True |
| `storage_backend` | Storage backend for saving checkpoints. | "local" |
| `sync_to_storage` | Whether to sync checkpoints to remote storage. | False |
| `load_from_storage` | Whether to load checkpoints from remote storage. | False |
| `remote_path` | Remote storage path (if applicable). | "" |
| **Reproducibility Settings** |
| `seed` | Random seed for reproducibility. | 1 |
| `deterministic` | Whether to enforce deterministic behavior. | True |
| **Other Settings** |
| `framework` | Training framework. | "pytorch" |
| `use_tt` | Whether to run on TT device (or CPU otherwise). | False |
