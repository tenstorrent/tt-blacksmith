# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time
import traceback
from pathlib import Path

import ale_py
import gymnasium as gym
import torch
import torch_xla

from blacksmith.experiments.torch.BOUNTIES.ppo_breakout.breakout_rollout import (
    RolloutBuffer,
)
from blacksmith.experiments.torch.BOUNTIES.ppo_breakout.configs import TrainingConfig
from blacksmith.models.torch.BOUNTIES.breakout_cnn import BreakoutCNN
from blacksmith.tools.checkpoints_manager import CheckpointManager
from blacksmith.tools.cli import generate_config, parse_cli_options
from blacksmith.tools.device_manager import DeviceManager
from blacksmith.tools.logging_manager import TrainingLogger
from blacksmith.tools.reproducibility_manager import ReproducibilityManager

gym.register_envs(ale_py)

# Small epsilon added to standard deviation when normalizing advantages, to avoid division by zero
ADV_NORM_EPS = 1e-8

# ---------------------------------------------------------------------------
# Environment wrappers
# ---------------------------------------------------------------------------


class EpisodicLifeEnv(gym.Wrapper):
    # Signal done on life loss without actually resetting the game

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    # Press FIRE after reset to launch the ball

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardEnv(gym.RewardWrapper):
    # Clip rewards to {-1, 0, +1}

    def reward(self, reward):
        return float((reward > 0) - (reward < 0))


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


def make_env(idx: int, config: TrainingConfig):
    def thunk():
        # The original Atari hardware cannot render all the sprites at once, so we let the
        # environment render every frame (frameskip=1) and have AtariPreprocessing skip
        # config.frame_skip frames, max-pooling over the last 2 to see all sprites on screen.
        env = gym.make(config.env_id, frameskip=1)
        env = gym.wrappers.RecordEpisodeStatistics(env)  # Track episode return and length for logging
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,  # Random no-ops on reset to add stochasticity to starting states
            frame_skip=config.frame_skip,  # Repeat each action for N frames, max-pooling last 2 to avoid flickering
            screen_size=84,  # Downscale to 84x84 to reduce input dimensionality
            terminal_on_life_loss=False,  # Handled by EpisodicLifeEnv wrapper instead
            grayscale_obs=True,  # Convert RGB to single channel, reducing input size by 3x
            scale_obs=True,  # Scale pixel values to [0, 1] float range
        )
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.FrameStackObservation(
            env, config.frame_stack
        )  # Stack N consecutive frames to give the agent temporal context
        env.action_space.seed(config.seed + idx)
        env.observation_space.seed(config.seed + idx)
        return env

    return thunk


def make_vec_env(config: TrainingConfig):
    envs = gym.vector.SyncVectorEnv([make_env(i, config) for i in range(config.num_envs)])
    return envs


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


def ppo_update(agent, optimizer, buffer, advantages, returns, config: TrainingConfig, device_manager: DeviceManager):
    b_obs, b_actions, b_log_probs, b_values = buffer.flatten()
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)

    clip_fracs = []

    for _ in range(config.update_epochs):
        perm = torch.randperm(config.batch_size).to(device_manager.device)
        for start in range(0, config.batch_size, config.minibatch_size):
            mb_inds = perm[start : start + config.minibatch_size]
            mb_obs = b_obs.index_select(0, mb_inds)
            mb_actions = b_actions.index_select(0, mb_inds)
            mb_log_probs = b_log_probs.index_select(0, mb_inds)
            mb_values = b_values.index_select(0, mb_inds)
            mb_adv = b_advantages.index_select(0, mb_inds)
            mb_returns = b_returns.index_select(0, mb_inds)

            _, new_log_prob, entropy, new_value = agent.get_action_and_value(mb_obs, mb_actions)
            log_ratio = new_log_prob - mb_log_probs
            ratio = log_ratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clip_fracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean())

            if config.norm_adv:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + ADV_NORM_EPS)

            # Policy loss
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            new_value = new_value.view(-1)
            if config.clip_vloss:
                v_clipped = mb_values + torch.clamp(new_value - mb_values, -config.clip_coef, config.clip_coef)
                v_loss1 = (new_value - mb_returns) ** 2
                v_loss2 = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
            else:
                v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            if config.use_tt:
                torch_xla.sync(wait=True)
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            device_manager.optimizer_step(optimizer)

    # Move all metrics to CPU for logging to avoid XLA sync issues.
    if config.use_tt:
        torch_xla.sync(wait=True)

    mean_clip_frac = torch.tensor(clip_fracs).mean().item() if clip_fracs else 0.0
    return (
        pg_loss.item(),
        v_loss.item(),
        entropy_loss.item(),
        approx_kl.item(),
        mean_clip_frac,
    )


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------


def collect_rollout(agent, envs, buffer, obs, done, config, device, episode_returns, global_step):
    for _ in range(config.num_steps):
        global_step += config.num_envs

        with torch.no_grad():
            action, log_prob, _, value = agent.get_action_and_value(obs)

        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())

        buffer.insert(
            obs,
            action,
            log_prob,
            torch.tensor(reward, device=device),
            done,
            value.flatten(),
        )

        obs = torch.tensor(next_obs, device=device)
        done = torch.as_tensor(terminated | truncated, dtype=torch.float32).to(device)

        if "_episode" not in infos:
            continue
        for i, finished in enumerate(infos["_episode"]):
            if finished:
                episode_returns.append(infos["episode"]["r"][i])

    return obs, done, global_step


def resume_from_checkpoint(config, checkpoint_manager, agent, optimizer, logger):
    start_update = 1
    global_step = 0
    if config.resume_from_checkpoint:
        checkpoint_info = checkpoint_manager.load_checkpoint(agent, optimizer)
        if checkpoint_info:
            start_update = checkpoint_info["step"] + 1
            global_step = checkpoint_info.get("metrics", {}).get("global_step", 0)
            logger.info(f"Resumed at update {start_update}, global_step {global_step:,}")
    return start_update, global_step


def train(
    config: TrainingConfig,
    device_manager: DeviceManager,
    logger: TrainingLogger,
    checkpoint_manager: CheckpointManager,
):
    device = device_manager.device

    envs = make_vec_env(config)
    num_actions = envs.single_action_space.n
    obs_shape = envs.single_observation_space.shape

    agent = BreakoutCNN(num_actions, config.frame_stack).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5, capturable=config.use_tt)

    logger.info(f"Model parameters: {sum(p.numel() for p in agent.parameters())}")

    # Resume from checkpoint if configured.
    start_update, global_step = resume_from_checkpoint(config, checkpoint_manager, agent, optimizer, logger)

    buffer = RolloutBuffer(config, obs_shape, device)

    # Start rollout collection.
    obs, _ = envs.reset(seed=config.seed)
    obs = torch.tensor(obs).to(device)
    done = torch.zeros(config.num_envs).to(device)

    start_time = time.time()
    episode_returns = []

    logger.info(f"Starting PPO training on {config.env_id}")
    logger.info(f"  Total timesteps : {config.total_timesteps:,}")
    logger.info(f"  Batch size      : {config.batch_size}")
    logger.info(f"  Minibatch size  : {config.minibatch_size}")
    logger.info(f"  Num updates     : {config.num_updates}")

    try:
        for update in range(start_update, config.num_updates + 1):
            # Anneal learning rate.
            if config.anneal_lr:
                frac = 1.0 - (update - 1) / config.num_updates
                optimizer.param_groups[0]["lr"] = config.learning_rate * frac

            # Collect rollout.
            obs, done, global_step = collect_rollout(
                agent, envs, buffer, obs, done, config, device, episode_returns, global_step
            )

            # Compute GAE.
            with torch.no_grad():
                next_value = agent.get_value(obs).flatten()
            advantages, returns = buffer.compute_gae(next_value, done)
            # Empirically faster on TT.
            if config.use_tt:
                torch_xla.sync()

            # PPO update.
            pg_loss, v_loss, ent_loss, approx_kl, clip_frac = ppo_update(
                agent, optimizer, buffer, advantages, returns, config, device_manager
            )

            # Logging.
            if update % config.log_interval == 0:
                sps = int(global_step / (time.time() - start_time))
                avg_ret = torch.tensor(episode_returns[-10:]).mean().item() if episode_returns else 0.0

                logger.log_metrics(
                    {
                        "charts/avg_return": avg_ret,
                        "charts/sps": sps,
                        "losses/pg_loss": pg_loss,
                        "losses/vf_loss": v_loss,
                        "losses/entropy_loss": ent_loss,
                        "losses/approx_kl": approx_kl,
                        "losses/clip_frac": clip_frac,
                    },
                    commit=True,
                    step=global_step,
                )

            # Save checkpoint.
            if update % config.save_interval == 0:
                checkpoint_manager.save_checkpoint(
                    agent,
                    step=update,
                    epoch=0,
                    optimizer=optimizer,
                    metrics={"global_step": global_step, "charts/avg_return": avg_ret},
                )

        # Final save.
        avg_ret = torch.tensor(episode_returns[-10:]).mean().item() if episode_returns else 0.0
        checkpoint_manager.save_checkpoint(
            agent,
            step=config.num_updates,
            epoch=0,
            optimizer=optimizer,
            metrics={"global_step": global_step, "charts/avg_return": avg_ret},
            checkpoint_name="agent_final.pt",
        )
        logger.info("Training finished successfully.")

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"Training failed with error: {str(e)}", traceback_str)
        raise
    finally:
        logger.finish()
        envs.close()


if __name__ == "__main__":
    # Config setup.
    default_config = Path(__file__).parent / "single_chip" / "ppo_breakout.yaml"
    args = parse_cli_options(default_config=default_config)
    config: TrainingConfig = generate_config(TrainingConfig, args.config)

    # Reproducibility setup.
    repro_manager = ReproducibilityManager(config)
    repro_manager.setup()

    # Logger setup.
    logger = TrainingLogger(config)

    # Checkpoint manager setup.
    checkpoint_manager = CheckpointManager(config, logger)

    # Device setup.
    device_manager = DeviceManager(config)
    logger.info(f"Using device: {device_manager.device}")

    # Start training.
    train(config, device_manager, logger, checkpoint_manager)
