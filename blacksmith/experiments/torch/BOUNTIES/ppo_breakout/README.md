# PPO Breakout Training

This directory contains the code for training a PPO (Proximal Policy Optimization) agent on the Atari Breakout game.
The agent uses a CNN architecture from `blacksmith/models/torch/BOUNTIES/breakout_cnn.py` with standard Atari preprocessing wrappers.


## Overview

The experiment trains a PPO agent on `ALE/Breakout-v5` using standard Atari preprocessing (frame skipping, grayscale, 84x84 resize, frame stacking) and environment wrappers (episodic life, fire reset, reward clipping). GAE (Generalized Advantage Estimation) is used for advantage computation.

The experiment is designed to run on TT hardware using the TT-XLA framework. For CPU baseline testing, set `use_tt: False` in the config file.

## Network Architecture

The agent uses the convolutional architecture introduced for Atari by Mnih et al. [1] — three
convolutional layers (32, 64, 64 channels) followed by a 512-unit fully-connected layer — with
separate actor and critic heads on top. Layers are orthogonally initialized: hidden layers use
a gain of √2, the policy (actor) head uses a gain of 0.01, and the value (critic) head uses a gain
of 1.0, following the PPO implementation details documented by Huang et al. [2]. The PPO objective
itself, along with the use of GAE for advantage estimation, is from Schulman et al. [3].

## Training

```bash
python blacksmith/experiments/torch/BOUNTIES/ppo_breakout/train.py
```

## Configuration

The experiment is configured using the configuration file `ppo_breakout.yaml`. Current defaults are the recommended and tested hyperparameters.

> **Note:** On TT, `num_minibatches` is set to 16 so the CNN's conv activations fit in L1; if you run on CPU (`use_tt: False`) you can use the standard value of 4.

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
| `num_minibatches` | Number of minibatches per update. | 16 |
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
| `use_tt` | Whether to run on TT device (or CPU otherwise). | True |

## References

[1] V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529–533, 2015.

[2] S. Huang, R. F. J. Dossa, A. Raffin, A. Kanervisto, and W. Wang, "The 37 Implementation Details of Proximal Policy Optimization," *ICLR Blog Track*, 2022.

[3] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," *arXiv:1707.06347*, 2017.
