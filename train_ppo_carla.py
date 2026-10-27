"""
PPO (Proximal Policy Optimization) training script for the CARLA driving task.

Provided alongside train_sac_carla.py (DDPG and TD3 as well, see
train_ddpg_carla.py / train_td3_carla.py) as a baseline comparison against
SAC, which is the algorithm actually used for the project's results — SAC
outperformed the others in practice, which is why it's the one with the full
training infrastructure (hyperparameter search, replay-buffer checkpointing,
Arduino hooks, etc.) in train_sac_carla.py.

This script reuses the exact same CarlaEnv and CombinedExtractor (CNN +
LIDAR CNN + state MLP feature fusion) as the SAC script, so the comparison
is apples-to-apples — only the RL algorithm differs.

Usage:
    python train_ppo_carla.py
    python train_ppo_carla.py --resume ./checkpoints_ppo/ppo_carla_20000_steps.zip
    python train_ppo_carla.py --total_timesteps 200000
"""

import os
import argparse

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from train_sac_carla import CombinedExtractor, make_env, console


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent on the CARLA driving task")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .zip to resume from")
    parser.add_argument("--total_timesteps", type=int, default=200_000, help="Total timesteps to train for")
    args = parser.parse_args()

    console.rule("[bold green]Starting PPO Training")

    env = DummyVecEnv([make_env() for _ in range(1)])

    # PPO's ActorCriticPolicy uses pi/vf (policy/value) heads rather than
    # SAC's pi/qf (policy/Q-function) — same feature extractor otherwise.
    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 512], vf=[256, 512]),
        share_features_extractor=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=2000, save_path="./checkpoints_ppo/", name_prefix="ppo_carla"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.resume is not None and os.path.exists(args.resume):
        console.log(f"[yellow]Resuming PPO training from checkpoint: {args.resume}[/yellow]")
        model = PPO.load(args.resume, env=env, device=device, custom_objects={"policy_kwargs": policy_kwargs})
    else:
        console.log("[cyan]Creating a new PPO model.[/cyan]")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="./ppo_tensorboard/",
            device=device,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
        )

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback,
                    reset_num_timesteps=args.resume is None)
    except KeyboardInterrupt:
        console.log("[bold yellow]Training interrupted by user (Ctrl+C). Saving model...[/bold yellow]")
    finally:
        model.save("ppo_carla_model")
        console.log("[bold green]Model saved to ppo_carla_model[/bold green]")
        try:
            env.close()
            console.log("[green]Environment closed successfully.[/green]")
        except Exception as e:
            console.log(f"[yellow]Warning during env cleanup: {e}[/yellow]")
