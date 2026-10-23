#!/usr/bin/env bash
# Run LOCALLY (git-bash). Copies the project + aws/ scripts to the instance
# and runs the one-time setup (CARLA download + venv). Re-running is safe;
# it just re-syncs files (CARLA download is skipped if already present).
set -euo pipefail

cd "$(dirname "$0")/.."
IP=$(cat aws/instance_ip.txt)
KEY="aws/carla-rl-key.pem"

ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@$IP" "mkdir -p ~/carla-rl/project/aws"

scp -i "$KEY" carla_env.py train_sac_carla.py "ubuntu@$IP:~/carla-rl/project/"
scp -i "$KEY" aws/sync_to_s3.sh aws/install_cron.sh aws/.env "ubuntu@$IP:~/carla-rl/project/aws/"
scp -i "$KEY" aws/remote_setup.sh aws/run_training.sh "ubuntu@$IP:~/carla-rl/"

echo "== Running remote setup (downloads CARLA + builds venv, ~15-20 min) =="
ssh -i "$KEY" "ubuntu@$IP" "chmod +x ~/carla-rl/remote_setup.sh ~/carla-rl/run_training.sh && ~/carla-rl/remote_setup.sh"

echo "== Installing S3 sync cron job =="
ssh -i "$KEY" "ubuntu@$IP" "~/carla-rl/project/aws/install_cron.sh"

echo "Done. Start training with: aws/connect.sh then ~/carla-rl/run_training.sh"
