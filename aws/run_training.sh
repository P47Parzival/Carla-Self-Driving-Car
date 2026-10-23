#!/usr/bin/env bash
# Runs ON the EC2 instance. Starts CARLA headless + training inside tmux so
# both survive an SSH disconnect. Re-run safely: it reuses the existing
# tmux session if one is already up.
set -euo pipefail

WORKDIR="$HOME/carla-rl"
CARLA_DIR="$WORKDIR/carla-server"
VENV_DIR="$WORKDIR/venv"
PROJECT_DIR="$WORKDIR/project"
SESSION="carla-rl"
TRAIN_ARGS="${1:-}"   # e.g. pass "--resume ./checkpoints/sac_carla_160000_steps.zip"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already running. Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n carla
tmux send-keys -t "$SESSION:carla" "cd $CARLA_DIR && ./CarlaUnreal.sh -RenderOffScreen -nosound -quality-level=Low" C-m

echo "Waiting 30s for CARLA server to come up..."
sleep 30

tmux new-window -t "$SESSION" -n train
tmux send-keys -t "$SESSION:train" "cd $PROJECT_DIR && source $VENV_DIR/bin/activate && python train_sac_carla.py $TRAIN_ARGS" C-m

tmux new-window -t "$SESSION" -n tensorboard
tmux send-keys -t "$SESSION:tensorboard" "cd $PROJECT_DIR && source $VENV_DIR/bin/activate && tensorboard --logdir ./sac_tensorboard --host 0.0.0.0" C-m

echo "Started. Attach with: tmux attach -t $SESSION"
echo "Switch windows inside tmux with: Ctrl+b then 0 (carla) / 1 (train) / 2 (tensorboard)"
echo "Detach without killing: Ctrl+b then d"
echo "View TensorBoard from your laptop via aws/connect.sh -> http://localhost:6006"
