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

if [ -z "$TRAIN_ARGS" ]; then
  LATEST_CKPT=$(ls -t "$PROJECT_DIR"/checkpoints/*.zip 2>/dev/null | head -1 || true)
  if [ -n "$LATEST_CKPT" ]; then
    echo "No --resume given; found existing checkpoint, auto-resuming from: $LATEST_CKPT"
    TRAIN_ARGS="--resume $LATEST_CKPT"
  else
    echo "No existing checkpoint found; starting fresh."
  fi
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already running. Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n carla
tmux send-keys -t "$SESSION:carla" "cd $CARLA_DIR && ./CarlaUnreal.sh -RenderOffScreen -nosound -quality-level=Low" C-m

echo "Waiting for CARLA server to open port 2000 (up to 5 min)..."
for i in $(seq 1 60); do
  if (echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null; then
    echo "Port open after ${i}x5s. Giving it 15s more to finish loading the map..."
    sleep 15
    break
  fi
  sleep 5
done

tmux new-window -t "$SESSION" -n train
tmux send-keys -t "$SESSION:train" "cd $PROJECT_DIR && source $VENV_DIR/bin/activate && env -u LD_LIBRARY_PATH python train_sac_carla.py $TRAIN_ARGS" C-m

tmux new-window -t "$SESSION" -n tensorboard
tmux send-keys -t "$SESSION:tensorboard" "cd $PROJECT_DIR && source $VENV_DIR/bin/activate && tensorboard --logdir ./sac_tensorboard --host 0.0.0.0" C-m

tmux new-window -t "$SESSION" -n live
tmux send-keys -t "$SESSION:live" "source $VENV_DIR/bin/activate && python $PROJECT_DIR/aws/live_view.py" C-m

tmux new-window -t "$SESSION" -n lidar
tmux send-keys -t "$SESSION:lidar" "source $VENV_DIR/bin/activate && python $PROJECT_DIR/aws/lidar_live_view.py" C-m

echo "Started. Attach with: tmux attach -t $SESSION"
echo "Switch windows inside tmux with: Ctrl+b then 0 (carla) / 1 (train) / 2 (tensorboard) / 3 (live) / 4 (lidar)"
echo "Detach without killing: Ctrl+b then d"
echo "View TensorBoard and the live feed from your laptop via aws/connect.sh"
