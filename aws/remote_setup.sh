#!/usr/bin/env bash
# Runs ON the EC2 instance. Installs CARLA 0.10.0 server + a dedicated Python
# venv with the training dependencies (no pyserial — software-only setup).
set -euo pipefail

CARLA_VERSION="0.10.0"
CARLA_URL="https://downloads.carlasim.com/Linux/Carla-0.10.0-Linux-Shipping.tar.gz"
WORKDIR="$HOME/carla-rl"
CARLA_DIR="$WORKDIR/carla-server"
VENV_DIR="$WORKDIR/venv"
PROJECT_DIR="$WORKDIR/project"

mkdir -p "$WORKDIR" "$CARLA_DIR" "$PROJECT_DIR/checkpoints" "$PROJECT_DIR/sac_tensorboard"

echo "== Checking NVIDIA driver =="
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader

echo "== Installing system dependencies =="
sudo apt-get update -y
sudo apt-get install -y libomp5 vulkan-tools libvulkan1 unzip tmux python3-venv python3-pip

echo "== Downloading CARLA $CARLA_VERSION server (this is a large download, ~15-20GB) =="
if [ ! -f "$WORKDIR/carla.tar.gz" ]; then
  curl -L -o "$WORKDIR/carla.tar.gz" "$CARLA_URL"
fi

echo "== Extracting CARLA =="
tar -xzf "$WORKDIR/carla.tar.gz" -C "$CARLA_DIR" --strip-components=1

echo "== Creating Python venv (isolated, not global) =="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo "== Installing training dependencies into venv =="
pip install stable-baselines3 gymnasium torch numpy opencv-python-headless pygame rich optuna tensorboard boto3

echo "== Installing the matching CARLA Python API wheel from the downloaded package =="
CARLA_WHEEL=$(find "$CARLA_DIR" -name "carla-*.whl" | head -n1)
if [ -z "$CARLA_WHEEL" ]; then
  echo "ERROR: could not find a carla-*.whl inside the extracted package. Check $CARLA_DIR/PythonAPI/"
  exit 1
fi
pip install "$CARLA_WHEEL"

echo "== Done. Venv at $VENV_DIR, CARLA server at $CARLA_DIR, project files go in $PROJECT_DIR =="
