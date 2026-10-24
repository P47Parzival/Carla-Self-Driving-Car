#!/usr/bin/env bash
# Runs ON the EC2 instance. Restarts just the training process in the existing
# tmux "train" window (use when training crashed but CARLA/TensorBoard/live
# view are still fine) — auto-resumes from the latest checkpoint unless you
# pass your own --resume/--total_timesteps args.
set -euo pipefail

PROJECT_DIR="$HOME/carla-rl/project"
VENV_DIR="$HOME/carla-rl/venv"
SESSION="carla-rl"
TRAIN_ARGS="${1:-}"

if [ -z "$TRAIN_ARGS" ]; then
  LATEST_CKPT=$(ls -t "$PROJECT_DIR"/checkpoints/*.zip 2>/dev/null | head -1 || true)
  if [ -n "$LATEST_CKPT" ]; then
    echo "Auto-resuming from: $LATEST_CKPT"
    TRAIN_ARGS="--resume $LATEST_CKPT"
  else
    echo "No checkpoint found; starting fresh."
  fi
fi

tmux send-keys -t "$SESSION:train" "cd $PROJECT_DIR && source $VENV_DIR/bin/activate && env -u LD_LIBRARY_PATH python train_sac_carla.py $TRAIN_ARGS" C-m
echo "Restart command sent to tmux window $SESSION:train"
