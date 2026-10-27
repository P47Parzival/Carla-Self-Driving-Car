"""
TD3 (Twin Delayed DDPG) training script for the CARLA driving task.

Provided alongside train_sac_carla.py (PPO and DDPG as well, see
train_ppo_carla.py / train_ddpg_carla.py) as a baseline comparison against
SAC, which is the algorithm actually used for the project's results — SAC
outperformed the others in practice, which is why it's the one with the full
training infrastructure (hyperparameter search, replay-buffer checkpointing,
Arduino hooks, etc.) in train_sac_carla.py.

This script reuses the exact same CarlaEnv and CombinedExtractor (CNN +
LIDAR CNN + state MLP feature fusion) as the SAC script, so the comparison
is apples-to-apples — only the RL algorithm differs.

Usage:
    python train_td3_carla.py
    python train_td3_carla.py --resume ./checkpoints_td3/td3_carla_20000_steps.zip
    python train_td3_carla.py --total_timesteps 200000
"""

import os
import argparse

import numpy as np
import torch

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise

from train_sac_carla import CombinedExtractor, make_env, console


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TD3 agent on the CARLA driving task")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .zip to resume from")
    parser.add_argument("--total_timesteps", type=int, default=200_000, help="Total timesteps to train for")
    args = parser.parse_args()

    console.rule("[bold green]Starting TD3 Training")

    env = DummyVecEnv([make_env() for _ in range(1)])

    # TD3, like DDPG, is a deterministic-policy algorithm and needs explicit
    # exploration noise (its "twin critics + delayed policy updates" improve
    # on DDPG's stability but don't change the exploration mechanism).
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 512], qf=[256, 512]),
        share_features_extractor=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=2000, save_path="./checkpoints_td3/", name_prefix="td3_carla"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.resume is not None and os.path.exists(args.resume):
        console.log(f"[yellow]Resuming TD3 training from checkpoint: {args.resume}[/yellow]")
        model = TD3.load(args.resume, env=env, device=device, custom_objects={"policy_kwargs": policy_kwargs})
    else:
        console.log("[cyan]Creating a new TD3 model.[/cyan]")
        model = TD3(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./td3_tensorboard/",
            device=device,
            learning_rate=1e-3,
            buffer_size=150_000,
            learning_starts=1000,
            batch_size=128,
            tau=0.005,
            gamma=0.99,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,
        )

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback,
                    reset_num_timesteps=args.resume is None)
    except KeyboardInterrupt:
        console.log("[bold yellow]Training interrupted by user (Ctrl+C). Saving model...[/bold yellow]")
    finally:
        model.save("td3_carla_model")
        console.log("[bold green]Model saved to td3_carla_model[/bold green]")
        try:
            env.close()
            console.log("[green]Environment closed successfully.[/green]")
        except Exception as e:
            console.log(f"[yellow]Warning during env cleanup: {e}[/yellow]")
