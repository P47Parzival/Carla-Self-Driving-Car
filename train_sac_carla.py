import os
import warnings

# ─── Suppress third-party deprecation noise ────────────────────────────
# These are from packages we don't control (pygame, tensorboard, protobuf).
# Our own code uses the latest APIs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # Silence TensorFlow INFO/WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"         # Suppress oneDNN message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"    # Suppress pygame hello banner
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
# ────────────────────────────────────────────────────────────────────────

import argparse

import gymnasium

import torch

import torch.nn as nn

import numpy as np

from stable_baselines3 import SAC

from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rich.console import Console

import optuna

from stable_baselines3.common.logger import configure

# Note: Mixed Precision Training is handled in carla_env.py's CustomSAC class
    
from carla_env import CarlaEnv, CustomSAC, _SERIAL_AVAILABLE



console = Console()



# NEW: Configure a logger that outputs to stdout, CSV, and tensorboard.

new_logger = configure("./sac_tensorboard/", ["stdout", "csv", "tensorboard"])


# Learning Rate Schedule Callback - decays LR from 3e-4 to 1.5e-4 over 300k steps
class FixedScheduleLRCallback(BaseCallback):
    """
    Decays learning rate from initial to final over a fixed number of steps.
    
    LR Schedule: 3e-4 → 1.5e-4 over 300k steps
    After 300k steps, LR stays constant at 1.5e-4.
    
    This is independent of total_timesteps passed to learn().
    Works for infinite training!
    
    Note: Only updates actor and critic LR. Entropy coefficient uses its own
    optimizer with separate LR (auto-tuned based on entropy objective).
    """
    
    def __init__(self, initial_lr: float = 3e-4, final_lr: float = 1.5e-4, 
                 schedule_steps: int = 300_000, log_interval: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.schedule_steps = schedule_steps
        self.log_interval = log_interval
    
    def _on_step(self) -> bool:
        # Calculate progress (capped at 1.0 for steps beyond schedule)
        progress = min(1.0, self.num_timesteps / self.schedule_steps)
        
        # Linear interpolation: initial_lr → final_lr
        new_lr = self.initial_lr - (self.initial_lr - self.final_lr) * progress
        
        # Update actor optimizer LR
        for param_group in self.model.actor.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Update critic optimizer LR
        for param_group in self.model.critic.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        # Note: Entropy coefficient optimizer (ent_coef_optimizer) is NOT updated
        # It uses its own LR and auto-tunes based on entropy objective
        
        # Log LR periodically
        if self.verbose > 0 and self.num_timesteps % self.log_interval == 0:
            lr_progress_pct = progress * 100
            print(f"[LR Schedule] Step {self.num_timesteps}: LR = {new_lr:.6f} "
                  f"(progress: {lr_progress_pct:.1f}% of 300k)")
        
        # Record to TensorBoard
        if self.logger is not None:
            self.logger.record("train/learning_rate", new_lr)
        
        return True


class EntropyLoggingCallback(BaseCallback):

    def __init__(self, log_interval: int = 1000, verbose: int = 1):

        super(EntropyLoggingCallback, self).__init__(verbose)

        self.log_interval = log_interval



    def _on_step(self) -> bool:

        if self.n_calls % self.log_interval == 0:

            # Access entropy coefficient (check log_ent_coef FIRST for auto entropy mode)

            if hasattr(self.model, 'log_ent_coef') and self.model.log_ent_coef is not None:

                current_ent_coef = torch.exp(self.model.log_ent_coef).item()

            elif hasattr(self.model, 'ent_coef_tensor'):

                current_ent_coef = self.model.ent_coef_tensor.item()

            else:

                current_ent_coef = 0.0

            print(f"[INFO] Step {self.n_calls}: Entropy Coefficient = {current_ent_coef:.4f}")

        return True


class EnvMetricsCallback(BaseCallback):
    """
    Logs environment driving metrics emitted by CarlaEnv via the info dict.
    """

    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._reset_acc()

    def _reset_acc(self) -> None:
        self._n = 0
        self._sum_speed_mps = 0.0
        self._sum_distance_moved = 0.0
        self._sum_danger_occ = 0.0
        self._collisions = 0
        self._lane_invasions = 0
        self._stuck_events = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            self._n += 1
            self._sum_speed_mps += float(info.get("speed_mps", 0.0))
            self._sum_distance_moved += float(info.get("distance_moved_m", 0.0))
            self._sum_danger_occ += float(info.get("danger_occupancy", 0.0))
            self._collisions += int(bool(info.get("collision", False)))
            self._lane_invasions += int(bool(info.get("lane_invasion", False)))
            self._stuck_events += int(bool(info.get("stuck", False)))

            ep = info.get("episode")
            if isinstance(ep, dict) and self.logger is not None:
                if "r" in ep:
                    self.logger.record("episode/return", float(ep["r"]))
                if "l" in ep:
                    self.logger.record("episode/length", int(ep["l"]))
                if "distance_m" in ep:
                    self.logger.record("episode/distance_m", float(ep["distance_m"]))
                if "lane_invasions" in ep:
                    self.logger.record("episode/lane_invasions", int(ep["lane_invasions"]))
                if "stuck_respawns" in ep:
                    self.logger.record("episode/stuck_respawns", int(ep["stuck_respawns"]))

        if self.n_calls % self.log_interval == 0 and self._n > 0 and self.logger is not None:
            n = float(self._n)
            avg_speed = self._sum_speed_mps / n
            self.logger.record("env/avg_speed_mps", avg_speed)
            self.logger.record("env/avg_speed_kmh", avg_speed * 3.6)
            self.logger.record("env/avg_distance_moved_m", self._sum_distance_moved / n)
            self.logger.record("env/avg_danger_occupancy", self._sum_danger_occ / n)
            self.logger.record("env/collision_rate", self._collisions / n)
            self.logger.record("env/lane_invasion_rate", self._lane_invasions / n)
            self.logger.record("env/stuck_event_rate", self._stuck_events / n)
            self._reset_acc()

        return True


# Replay Buffer Checkpoint Callback - saves full checkpoint every 10k steps
class ReplayBufferCheckpointCallback(BaseCallback):
    """
    Saves replay buffer alongside model checkpoints.
    
    This enables seamless resume of training without losing experience.
    The replay buffer is saved with LZMA compression to reduce file size
    from ~3-4 GB to ~1-1.5 GB.
    
    Automatically cleans up old checkpoints, keeping only the last N (default: 3).
    This prevents disk space from filling up with large replay buffer files.
    
    Args:
        save_freq: How often to save (default: 10000 steps)
        save_path: Directory to save checkpoints
        name_prefix: Prefix for checkpoint files
        keep_last_n: Number of checkpoints to keep (default: 3)
        verbose: Verbosity level
    """
    
    def __init__(self, save_freq: int = 10000, save_path: str = './checkpoints/', 
                 name_prefix: str = 'sac_carla', keep_last_n: int = 3, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.keep_last_n = keep_last_n
        self.last_save_step = 0
        self.saved_checkpoints = []  # Track saved checkpoint steps for cleanup
    
    def _init_callback(self) -> None:
        # Create save directory if it doesn't exist
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        
        # Scan for existing checkpoints to populate saved_checkpoints list
        self._scan_existing_checkpoints()
    
    def _scan_existing_checkpoints(self) -> None:
        """Scan checkpoint directory for existing checkpoints."""
        import re
        import glob
        
        # Pattern to match checkpoint files: sac_carla_12345_steps.zip
        pattern = os.path.join(self.save_path, f"{self.name_prefix}_*_steps.zip")
        existing_files = glob.glob(pattern)
        
        # Extract step numbers from filenames
        step_pattern = re.compile(rf"{re.escape(self.name_prefix)}_(\d+)_steps\.zip")
        steps = []
        for filepath in existing_files:
            filename = os.path.basename(filepath)
            match = step_pattern.match(filename)
            if match:
                steps.append(int(match.group(1)))
        
        # Sort and store
        self.saved_checkpoints = sorted(steps)
        if self.saved_checkpoints and self.verbose > 0:
            console.log(f"[cyan]Found {len(self.saved_checkpoints)} existing checkpoints: {self.saved_checkpoints}[/cyan]")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        if len(self.saved_checkpoints) <= self.keep_last_n:
            return
        
        # Checkpoints to remove (oldest ones)
        to_remove = self.saved_checkpoints[:-self.keep_last_n]
        
        for step in to_remove:
            # Remove model checkpoint (.zip)
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{step}_steps.zip")
            # Remove replay buffer checkpoint (.pkl.xz)
            buffer_path = os.path.join(self.save_path, f"{self.name_prefix}_{step}_steps_replay_buffer.pkl.xz")
            # Also try uncompressed buffer path (.pkl)
            buffer_path_uncompressed = os.path.join(self.save_path, f"{self.name_prefix}_{step}_steps_replay_buffer.pkl")
            
            removed_files = []
            for path in [model_path, buffer_path, buffer_path_uncompressed]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        removed_files.append(os.path.basename(path))
                    except Exception as e:
                        console.log(f"[yellow]Warning: Could not remove {path}: {e}[/yellow]")
            
            if removed_files and self.verbose > 0:
                console.log(f"[yellow]Cleaned up old checkpoint (step {step:,}): {', '.join(removed_files)}[/yellow]")
        
        # Update the list
        self.saved_checkpoints = self.saved_checkpoints[-self.keep_last_n:]
    
    def _on_step(self) -> bool:
        # Only save at save_freq intervals
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            self.last_save_step = self.num_timesteps
            
            # Construct paths
            checkpoint_path = os.path.join(
                self.save_path, 
                f"{self.name_prefix}_{self.num_timesteps}_steps"
            )
            replay_buffer_path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps_replay_buffer"
            )
            
            # Save model (weights, hyperparams)
            self.model.save(checkpoint_path)
            
            # Save replay buffer with compression
            if hasattr(self.model, 'save_replay_buffer'):
                try:
                    self.model.save_replay_buffer(replay_buffer_path, use_compression=True)
                    
                    # Track this checkpoint
                    self.saved_checkpoints.append(self.num_timesteps)
                    
                    if self.verbose > 0:
                        console.log(f"[bold green]Full checkpoint saved at step {self.num_timesteps:,}[/bold green]")
                        console.log(f"[green]  Model: {checkpoint_path}.zip[/green]")
                        console.log(f"[green]  Replay Buffer: {replay_buffer_path}.pkl.xz[/green]")
                    
                    # Cleanup old checkpoints (keep last N)
                    self._cleanup_old_checkpoints()
                    
                except Exception as e:
                    console.log(f"[red]Error saving replay buffer: {e}[/red]")
            else:
                if self.verbose > 0:
                    console.log(f"[yellow]Model saved but replay buffer save not available[/yellow]")
        
        return True
    
    def _on_training_start(self) -> None:
        """Initialize last_save_step based on current timesteps (for resume)."""
        if hasattr(self.model, 'num_timesteps'):
            # Round down to nearest save_freq to avoid double-saving
            self.last_save_step = (self.model.num_timesteps // self.save_freq) * self.save_freq





# --- Define a Residual Block for the CNN ---

class ResidualBlock(nn.Module):

    def __init__(self, channels: int):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

        

    def forward(self, x):

        residual = x

        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        return self.relu(out + residual)



# --- Enhanced Feature Extractor with Attention Fusion ---

class CombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gymnasium.spaces.Dict, features_dim: int = 1024):

        # State dimension is read dynamically from observation_space

        super(CombinedExtractor, self).__init__(observation_space, features_dim)
        
        # Get image dimensions from observation space
        # NOTE: SB3's VecTransposeImage wrapper transposes the obs space to (C, H, W)
        # So we need to detect if it's channel-first or channel-last
        image_shape = observation_space.spaces["image"].shape
        
        # Detect format: compare first and last dims.
        # Channel-first (C, H, W): first dim is 1-3, last dim is spatial (≥ first)
        # Channel-last (H, W, C): last dim is 1-3, first dim is spatial (≥ last)
        # Robust: if first dim < last dim AND first dim ≤ 3 → channel-first
        if image_shape[0] <= 3 and image_shape[0] < image_shape[2]:
            # Channel-first format: (C, H, W) - after VecTransposeImage
            image_channels, image_height, image_width = image_shape
        else:
            # Channel-last format: (H, W, C) - original gymnasium format
            image_height, image_width, image_channels = image_shape
        
        console.log(f"[cyan]Image observation shape: {image_shape} -> C={image_channels}, H={image_height}, W={image_width}[/cyan]")
        
        # ========== IMAGE CNN ==========
        # Updated for GRAYSCALE input (1 channel instead of 3)
        # Input: 120x160x1 grayscale for better lane detection
        # Uses Global Average Pooling to fix output size regardless of input resolution
        # This prevents FC layer explosion when using higher resolutions
        self.cnn = nn.Sequential(

            nn.Conv2d(image_channels, 64, kernel_size=8, stride=4),  # Dynamic channel count

            nn.ReLU(),

            ResidualBlock(64),

            nn.Conv2d(64, 128, kernel_size=4, stride=2),

            nn.ReLU(),

            ResidualBlock(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),

            nn.ReLU(),

            ResidualBlock(256),

            nn.Conv2d(256, 256, kernel_size=3, stride=1),

            nn.ReLU(),

            # Global Average Pooling: reduces any spatial size to (256, 4, 4)
            # This fixes the FC layer size regardless of input resolution!
            # 120x160 or 84x84 → always outputs 256*4*4 = 4096 features
            nn.AdaptiveAvgPool2d((4, 4)),

            nn.Flatten()

        )

        # Verify CNN output dimension with a dummy tensor
        # This catches any bugs in the architecture or channel format detection
        with torch.no_grad():
            dummy_image = torch.zeros(1, image_channels, image_height, image_width)
            actual_cnn_out = self.cnn(dummy_image)
            cnn_out_dim = actual_cnn_out.shape[1]
        
        expected_cnn_out = 256 * 4 * 4  # = 4096 (from AdaptiveAvgPool2d((4, 4)))
        
        if cnn_out_dim != expected_cnn_out:
            console.log(f"[red][WARNING] CNN output mismatch! Expected {expected_cnn_out}, got {cnn_out_dim}[/red]")
            console.log(f"[red]Input shape: (1, {image_channels}, {image_height}, {image_width})[/red]")
            console.log(f"[red]Output shape: {actual_cnn_out.shape}[/red]")
        
        console.log(f"[green]CNN output dimension: {cnn_out_dim} (verified with dummy tensor)[/green]")
        
        # ========== LIDAR BEV CNN ==========
        # Processes the 64x64 single-channel BEV occupancy grid
        # NOTE: Same format detection as image - VecTransposeImage affects all image-like observations
        lidar_shape = observation_space.spaces["lidar_bev"].shape
        
        # Detect format: same robust heuristic as image
        if lidar_shape[0] <= 3 and lidar_shape[0] < lidar_shape[2]:
            # Channel-first format: (C, H, W) - after VecTransposeImage
            lidar_channels, lidar_height, lidar_width = lidar_shape
        else:
            # Channel-last format: (H, W, C) - original gymnasium format
            lidar_height, lidar_width, lidar_channels = lidar_shape
        
        console.log(f"[cyan]LIDAR BEV observation shape: {lidar_shape} -> C={lidar_channels}, H={lidar_height}, W={lidar_width}[/cyan]")
        
        self.lidar_cnn = nn.Sequential(
            nn.Conv2d(lidar_channels, 32, kernel_size=5, stride=2),    # 64x64 -> 30x30
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),   # 30x30 -> 14x14
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),   # 14x14 -> 6x6
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Determine LIDAR CNN output dimension
        with torch.no_grad():
            dummy_lidar = torch.zeros(1, lidar_channels, lidar_height, lidar_width)
            lidar_cnn_out_dim = self.lidar_cnn(dummy_lidar).shape[1]
        
        # ========== STATE MLP ==========
        state_dim = observation_space.spaces["state"].shape[0]  # dynamically determined (currently 4)
        # State features: 60 (~2.9% of total 2048)
        state_features_dim = 60
        
        # MLP for state: maps 4D state to 60 features
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_features_dim),
            nn.ReLU()
        )

        
        # ========== FUSION LAYERS (CAMERA-BIASED: 54% Camera, 43% LIDAR, 3% State) ==========
        # 
        # Feature distribution optimized for lane following with camera!
        # 
        # Total: 2048 features
        # - Camera:  1106 features (54.0%) - lane following, road texture, visual cues
        # - LIDAR:    882 features (43.1%) - obstacle detection, spatial awareness
        # - State:     60 features (2.9%)  - speed, position, heading
        # 
        # Camera boosted for better lane detection while maintaining solid LIDAR coverage
        
        # Camera bottleneck: compress 4096 → 1106 features (54%)
        camera_features_dim = 1106
        self.camera_bottleneck = nn.Sequential(
            nn.Linear(cnn_out_dim, camera_features_dim),
            nn.ReLU()
        )
        
        # Attention layer: camera (1106) guides state importance (60 features)
        self.attention_layer = nn.Linear(camera_features_dim, state_features_dim)
        
        # LIDAR feature projection: project to 882 features (43.1%)
        # Still substantial for obstacle avoidance!
        lidar_features_dim = 882
        self.lidar_projection = nn.Sequential(
            nn.Linear(lidar_cnn_out_dim, 256),
            nn.ReLU(),
            nn.Linear(256, lidar_features_dim),
            nn.ReLU()
        )
        
        # Combined: camera + state + lidar = 1106 + 60 + 882 = 2048
        combined_dim = camera_features_dim + state_features_dim + lidar_features_dim

        # Log the camera-biased feature distribution
        console.log(f"[bold green]╔═══════════════════════════════════════════════════════╗[/bold green]")
        console.log(f"[bold green]║   FEATURE DISTRIBUTION (54% Cam, 43% LIDAR, 3% State) ║[/bold green]")
        console.log(f"[bold green]╠═══════════════════════════════════════════════════════╣[/bold green]")
        console.log(f"[green]║ Camera:  {camera_features_dim:>5} features ({100*camera_features_dim/combined_dim:.1f}%) - BOOSTED![/green]")
        console.log(f"[green]║ LIDAR:   {lidar_features_dim:>5} features ({100*lidar_features_dim/combined_dim:.1f}%) - projection[/green]")
        console.log(f"[green]║ State:   {state_features_dim:>5} features ({100*state_features_dim/combined_dim:.1f}%) - MLP (4 scalars)[/green]")
        console.log(f"[bold green]╠═══════════════════════════════════════════════════════╣[/bold green]")
        console.log(f"[bold cyan]║ TOTAL:   {combined_dim:>5} features (100.0%)[/bold cyan]")
        console.log(f"[bold green]║ Camera boosted for better lane following!             ║[/bold green]")
        console.log(f"[bold green]╚═══════════════════════════════════════════════════════╝[/bold green]")

        # Two-stage FC to reduce parameters: 2048 -> 512 -> features_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

        self._features_dim = features_dim

        

        # Initialize weights using Kaiming/He initialization (best practice for ReLU networks)

        self._initialize_weights()
        
        # Log parameter count for debugging VRAM usage
        total_params = sum(p.numel() for p in self.parameters())
        cnn_params = sum(p.numel() for p in self.cnn.parameters())
        lidar_params = sum(p.numel() for p in self.lidar_cnn.parameters())
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        attn_params = sum(p.numel() for p in self.attention_layer.parameters())
        lidar_proj_params = sum(p.numel() for p in self.lidar_projection.parameters())
        fc_params = sum(p.numel() for p in self.fc.parameters())
        
        console.log(f"[bold yellow]╔═══════════════════════════════════════════════════════╗[/bold yellow]")
        console.log(f"[bold yellow]║      FEATURE EXTRACTOR PARAMETER COUNT                ║[/bold yellow]")
        console.log(f"[bold yellow]╠═══════════════════════════════════════════════════════╣[/bold yellow]")
        console.log(f"[yellow]║ Image CNN:        {cnn_params:>10,} params[/yellow]")
        console.log(f"[yellow]║ LIDAR CNN:        {lidar_params:>10,} params[/yellow]")
        console.log(f"[yellow]║ State MLP:        {mlp_params:>10,} params[/yellow]")
        console.log(f"[yellow]║ Attention Layer:  {attn_params:>10,} params[/yellow]")
        console.log(f"[yellow]║ LIDAR Projection: {lidar_proj_params:>10,} params[/yellow]")
        console.log(f"[yellow]║ FC Layer:         {fc_params:>10,} params[/yellow]")
        console.log(f"[bold yellow]╠═══════════════════════════════════════════════════════╣[/bold yellow]")
        console.log(f"[bold green]║ TOTAL:            {total_params:>10,} params[/bold green]")
        console.log(f"[bold yellow]╚═══════════════════════════════════════════════════════╝[/bold yellow]")
        
        # VRAM estimation (FP32: 4 bytes per param, FP16: 2 bytes)
        vram_fp32_mb = (total_params * 4) / (1024 * 1024)
        vram_fp16_mb = (total_params * 2) / (1024 * 1024)
        console.log(f"[cyan]Estimated VRAM (weights only): {vram_fp32_mb:.1f} MB (FP32) / {vram_fp16_mb:.1f} MB (FP16)[/cyan]")

    

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:

                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                nn.init.constant_(m.bias, 0)



    def forward(self, observations):
        # ========== CAMERA-BIASED FEATURE EXTRACTION ==========
        # Feature distribution: Camera 54% (1106), LIDAR 43% (882), State 3% (60)
        # Total: 2048 features optimized for lane following!
        
        # Process image (no autocast here - causes dtype issues with fc layer)
        # SB3 preprocess_obs() already scales image Box observations to [0, 1].
        # Do not divide by 255 here again or camera/LIDAR signals become ~255x too small.
        # Handle both channel-first (batch, C, H, W) and channel-last (batch, H, W, C) formats
        # For grayscale: C=1, for RGB: C=3
        img = observations["image"]
        if img.ndim == 4:
            # Detect channel format: channel-first if dim1 ≤ 3 AND dim1 < dim3
            if img.shape[1] <= 3 and img.shape[1] < img.shape[3]:
                # Already channel-first format (batch, C, H, W)
                image = img.float()
            else:
                # Channel-last format (batch, H, W, C), permute to (batch, C, H, W)
                image = img.permute(0, 3, 1, 2).float()
        else:
            # Fallback: assume channel-last and permute
            image = img.permute(0, 3, 1, 2).float()
        
        # Camera CNN → bottleneck (4096 → 1106 features, 54%)
        raw_image_features = self.cnn(image)  # shape: [batch, 4096]
        image_features = self.camera_bottleneck(raw_image_features)  # shape: [batch, 1106]
        
        # Process LIDAR BEV grid. SB3 has already normalized image-like uint8 spaces.
        # LIDAR BEV is 64x64x1, same handling as image
        lidar_bev = observations["lidar_bev"]
        if lidar_bev.ndim == 4:
            # Detect channel format: channel-first if dim1 ≤ 3 AND dim1 < dim3
            if lidar_bev.shape[1] <= 3 and lidar_bev.shape[1] < lidar_bev.shape[3]:
                # Already channel-first format (batch, 1, 64, 64)
                lidar = lidar_bev.float()
            else:
                # Channel-last format (batch, 64, 64, 1), permute to (batch, 1, 64, 64)
                lidar = lidar_bev.permute(0, 3, 1, 2).float()
        else:
            # Fallback: assume channel-last and permute
            lidar = lidar_bev.permute(0, 3, 1, 2).float()
        
        # LIDAR CNN → projection (raw → 882 features, 43%)
        raw_lidar_features = self.lidar_cnn(lidar)  # shape: [batch, lidar_cnn_out_dim]
        lidar_features = self.lidar_projection(raw_lidar_features)  # shape: [batch, 882]
        
        # Process state (60 features, ~3%)
        state_features = self.mlp(observations["state"].float())  # shape: [batch, 60]
        
        # Attention-weighted fusion (bottlenecked camera 1106 → guides state 60)
        attn_weights = torch.sigmoid(self.attention_layer(image_features))  # shape: [batch, 60]
        fused_state = state_features * attn_weights
        
        # Concatenate all features: camera + state (attention-weighted) + lidar
        # Total: 1106 + 60 + 882 = 2048 features (Camera-biased for lane following!)
        concatenated = torch.cat([image_features, fused_state, lidar_features], dim=1)

        concatenated = torch.nan_to_num(concatenated, nan=0.0, posinf=1e3, neginf=-1e3)
        return self.fc(concatenated)







def make_env():

    def _init():

        # Try to find the correct Arduino port (guarded: works without pyserial)
        arduino_port = 'COM3'  # Default fallback
        if _SERIAL_AVAILABLE:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            
            for port in ports:
                if "Arduino" in port.description or "USB Serial Device" in port.description:
                    arduino_port = port.device
                    break
            else:
                console.log("[yellow]No Arduino found. Available ports:[/yellow]")
                for port in ports:
                    console.log(f"[yellow]{port.device}: {port.description}[/yellow]")
        else:
            console.log("[yellow]pyserial not installed — using default port (Arduino disabled)[/yellow]")
            
        console.log(f"[green]Using Arduino port: {arduino_port}[/green]")
        
        # ========== PERFORMANCE TIP ==========
        # DO NOT use no_rendering_mode=True! It disables ALL GPU sensors (cameras return black).
        # Instead, launch CARLA with: CarlaUE4.exe -RenderOffScreen
        # This hides the CARLA 3D viewport but keeps sensors working (2-3x speedup).
        # 
        # GRAYSCALE 128x128: Higher resolution for better visual detail
        # - 128x128x1 = 16,384 bytes (2.3x more pixels than 84x84)
        # - Grayscale preserves edge information (lane lines are high contrast)
        # - Compensated by reducing batch_size (512→256) and buffer (50k→35k)
        env = CarlaEnv(
            num_npcs=5, 
            frame_skip=8, 
            visualize=True,  # This is pygame window (your dashboard)
            fixed_delta_seconds=0.05,
            camera_width=128,    # Higher resolution for better detail
            camera_height=128,   # Square 128x128 (power of 2 for CNNs)
            model=None,
            arduino_port=arduino_port,
            rotate_maps=False,  # Disabled: only 1 map available, enable when more maps downloaded
            no_rendering_mode=False  # MUST be False! True makes cameras return empty/black images
        )
        return env
    return _init





class CustomCheckpointCallback(CheckpointCallback):

    def _on_training_start(self) -> None:
        """SB3 already handles resume correctly with reset_num_timesteps=False.
        We no longer override n_calls which would break save_freq intervals."""
        super()._on_training_start()





def objective(trial):

    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)

    batch_size = trial.suggest_categorical('batch_size', [256, 512])

    tau = trial.suggest_float('tau', 0.001, 0.01)

    

    policy_kwargs = dict(

        features_extractor_class=CombinedExtractor,

        features_extractor_kwargs=dict(features_dim=256),  # Match main training (was 1024 — mismatch!)

        net_arch=dict(pi=[256, 512], qf=[256, 512]),  # Match main training (was [1024, 1024])
        
        # CRITICAL: Share feature extractor between actor and critics
        share_features_extractor=True

    )

    env = DummyVecEnv([make_env() for _ in range(1)])

    model = CustomSAC(

        "MultiInputPolicy",

        env,

        verbose=1,

        tensorboard_log="./sac_tensorboard/",

        device="cuda" if torch.cuda.is_available() else "cpu",

        learning_rate=lr,

        buffer_size=150000,

        # Note: optimize_memory_usage not supported with Dict observation spaces

        learning_starts=1000,

        batch_size=batch_size,

        tau=tau,

        policy_kwargs=policy_kwargs

    )



    # Train for a short trial.

    model.learn(total_timesteps=10000)

    # Evaluate performance over 1000 steps.
    # NOTE: DummyVecEnv uses the VecEnv API (4-tuple), NOT raw gymnasium (5-tuple).
    # VecEnv.step() returns (obs, rewards, dones, infos) where dones = terminated | truncated.
    # VecEnv.reset() returns just obs (no info dict).

    rewards = []

    obs = env.reset()

    for _ in range(1000):

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, _info = env.step(action)

        # DummyVecEnv returns arrays/lists even for a single env.
        if isinstance(reward, (list, tuple, np.ndarray)):
            rewards.append(float(reward[0]))
        else:
            rewards.append(float(reward))

        done_flag = bool(done[0]) if isinstance(done, (list, tuple, np.ndarray)) else bool(done)
        if done_flag:

            obs = env.reset()

    avg_reward = np.mean(rewards)

    # Clean up env to prevent CARLA actor leak between trials
    env.close()

    return avg_reward



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train SAC on CARLA environment with advanced features")

    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint/model file to resume training from")

    parser.add_argument("--total_timesteps", type=int, default=None, help="Total timesteps for training (default: infinite, use Ctrl+C to stop)")

    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization before training")

    args = parser.parse_args()



    if args.optimize:

        console.log("[bold yellow]Starting hyperparameter optimization...[/bold yellow]")

        study = optuna.create_study(direction='maximize')

        study.optimize(objective, n_trials=10)

        best_params = study.best_trial.params

        console.log(f"[bold green]Best hyperparameters: {best_params}[/bold green]")

        learning_rate = best_params['learning_rate']

        batch_size = best_params['batch_size']

        tau = best_params['tau']

    else:

        learning_rate = 2e-4  # Static LR (no decay schedule)

        batch_size = 128  # Reduced from 256 to save VRAM (~100-200 MB less activation memory)

        tau = 0.004



    console.rule("[bold green]Starting Training")

    env = DummyVecEnv([make_env() for _ in range(1)])



    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),  # Match paper's network size
        net_arch=dict(pi=[256, 512], qf=[256, 512]),  # Larger second layer for more capacity
        
        # CRITICAL: Share feature extractor between actor and critics
        # Without this, SB3 creates 3 separate extractors (1 actor + 2 critics)
        # This reduces VRAM by ~2/3 and speeds up training
        share_features_extractor=True
    )

    checkpoint_callback = CustomCheckpointCallback(save_freq=2000, save_path='./checkpoints/', name_prefix='sac_carla')



    class LossLoggingCallback(BaseCallback):

        def __init__(self, log_interval: int = 1000, verbose: int = 1):

            super(LossLoggingCallback, self).__init__(verbose)

            self.log_interval = log_interval



        def _on_step(self) -> bool:

            if self.n_calls % self.log_interval == 0:

                self.logger.record("custom/progress", self.n_calls)

                if self.verbose > 0:

                    print(f"Step: {self.n_calls}")

            return True



    class StuckDetectionCallback(BaseCallback):     

        def __init__(self, verbose: int = 1):

            super(StuckDetectionCallback, self).__init__(verbose)



        def _on_step(self) -> bool: 

            infos = self.locals.get("infos", [])

            for info in infos:

                if isinstance(info, dict) and info.get("stuck", False):

                    console.log("[yellow][STUCK CALLBACK] Episode ended because vehicle was stuck.[/yellow]")

            return True

        

    loss_logging_callback = LossLoggingCallback(log_interval=1000, verbose=1)

    stuck_detection_callback = StuckDetectionCallback(verbose=1)

    

    entropy_logging_callback = EntropyLoggingCallback(log_interval=1000, verbose=1)

    env_metrics_callback = EnvMetricsCallback(log_interval=1000, verbose=0)



    if args.resume is not None and os.path.exists(args.resume):

        console.log(f"[yellow]Resuming training from checkpoint: {args.resume}[/yellow]")

        
        # CRITICAL: Pass the same policy_kwargs with custom CombinedExtractor to load()
        # Without this, SB3 creates a default policy architecture that doesn't match the checkpoint!
        # This is the EXACT same policy_kwargs used when creating the model above.
        custom_objects = {
            "policy_kwargs": policy_kwargs,  # Use the policy_kwargs defined earlier with CombinedExtractor
        }
        
        # Load the model with custom objects to ensure correct architecture
        model = CustomSAC.load(

            args.resume, 

            env=env,

            device="cuda" if torch.cuda.is_available() else "cpu",
            custom_objects=custom_objects

        )   

        

        # Update the logger

        model.set_logger(new_logger)

        

        # Log the current state (check log_ent_coef first for auto entropy mode)

        if hasattr(model, 'log_ent_coef') and model.log_ent_coef is not None:

            current_alpha = torch.exp(model.log_ent_coef).item()

        else:

            current_alpha = model.ent_coef_tensor.item()

        console.log(f"[cyan]Current entropy coefficient: {current_alpha:.4f}[/cyan]")

        console.log(f"[cyan]Current timesteps: {model.num_timesteps}[/cyan]")

    else:

        console.log("[cyan]Creating a new model.[/cyan]")

        model = CustomSAC(

            "MultiInputPolicy",

            env,

            verbose=1,

            tensorboard_log="./sac_tensorboard/",

            device="cuda" if torch.cuda.is_available() else "cpu",

            learning_rate=learning_rate,

            buffer_size=150000,  # 150k transitions (~6GB RAM with 128x128 images + BEV)

            # Note: optimize_memory_usage not supported with Dict observation spaces

            learning_starts=1000,

            batch_size=batch_size,

            tau=tau,

            policy_kwargs=policy_kwargs

        )

        # Set the logger AFTER creation (SAC.__init__ doesn't accept a logger kwarg;
        # passing it is silently ignored via kwargs.pop in CustomSAC)
        model.set_logger(new_logger)



    console.log("[bold green]Starting training...")
    
    # Static LR: 2e-4 (no decay schedule)
    # LR schedule callback removed - using constant learning rate
    
    # Create replay buffer checkpoint callback (full checkpoint every 10k steps)
    # This saves both model weights AND replay buffer for seamless resume
    # Keeps only last 3 checkpoints to save disk space (~1.5GB each)
    replay_buffer_checkpoint_callback = ReplayBufferCheckpointCallback(
        save_freq=10000,  # Save every 10k steps (vs 2k for lightweight model-only saves)
        save_path='./checkpoints/',
        name_prefix='sac_carla_full',
        keep_last_n=3,  # Keep last 3 checkpoints (~4.5GB max), auto-delete older ones
        verbose=1
    )

    callbacks = [checkpoint_callback, loss_logging_callback, stuck_detection_callback, entropy_logging_callback, env_metrics_callback, replay_buffer_checkpoint_callback]

    # NOTE: model reference is accessed through the callback's self.model property,
    # NOT via a circular reference on the env (which causes memory leaks and pickle failures).
    
    # Determine total timesteps for training
    # If None (default), train infinitely until Ctrl+C
    # The LR schedule is independent - it decays over 300k then stays constant
    if args.total_timesteps is None:
        total_timesteps = int(1e9)  # ~1 billion steps (effectively infinite)
        console.log("[bold yellow]Training indefinitely (Ctrl+C to stop). Static LR = 2e-4.[/bold yellow]")
    else:
        total_timesteps = args.total_timesteps
        console.log(f"[cyan]Training for {total_timesteps:,} steps. Static LR = 2e-4.[/cyan]")

    # When resuming, continue from current timestep (don't reset counter)
    # Wrapped in try/finally so Ctrl+C still saves model and cleans up CARLA actors
    try:
        if args.resume is not None and os.path.exists(args.resume):
            if args.total_timesteps is not None:
                remaining_timesteps = args.total_timesteps - model.num_timesteps
                if remaining_timesteps <= 0:
                    console.log(f"[yellow]Model already trained for {model.num_timesteps} steps (target: {args.total_timesteps}). Nothing to do.[/yellow]")
                else:
                    console.log(f"[cyan]Resuming training for {remaining_timesteps:,} more steps...[/cyan]")
                    model.learn(total_timesteps=remaining_timesteps, callback=callbacks, reset_num_timesteps=False)
            else:
                # Infinite training when resuming (args.total_timesteps is None)
                console.log(f"[cyan]Resuming from step {model.num_timesteps:,}. Training indefinitely (Ctrl+C to stop)...[/cyan]")
                model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=False)
        else:
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
    except KeyboardInterrupt:
        console.log("[bold yellow]Training interrupted by user (Ctrl+C). Saving model...[/bold yellow]")
    finally:
        model_path = "sac_carla_model_enhanced"
        try:
            model.save(model_path)
            console.log(f"[bold green]Model saved to {model_path}[/bold green]")
        except Exception as e:
            console.log(f"[bold red]Failed to save model: {e}[/bold red]")
            # Fallback: save just the policy state_dict with PyTorch
            try:
                import torch
                torch.save(model.policy.state_dict(), model_path + "_policy_backup.pt")
                console.log(f"[yellow]Saved policy state_dict backup to {model_path}_policy_backup.pt[/yellow]")
            except Exception as e2:
                console.log(f"[red]Backup save also failed: {e2}[/red]")

        # Clean up CARLA environment (destroys actors, sensors, pygame)
        try:
            env.close()
            console.log("[green]Environment closed successfully.[/green]")
        except Exception as e:
            console.log(f"[yellow]Warning during env cleanup: {e}[/yellow]")
