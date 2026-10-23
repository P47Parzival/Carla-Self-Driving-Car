#!/usr/bin/env bash
# Runs ON the EC2 instance. Backs up checkpoints + tensorboard logs to S3 so
# progress survives a stopped/terminated instance. Reads AWS creds + bucket
# from ~/carla-rl/project/.env (see aws/.env.example).
set -euo pipefail

PROJECT_DIR="$HOME/carla-rl/project"
set -a
source "$PROJECT_DIR/.env"
set +a

aws s3 sync "$PROJECT_DIR/checkpoints" "s3://$S3_BUCKET/checkpoints" --only-show-errors
aws s3 sync "$PROJECT_DIR/sac_tensorboard" "s3://$S3_BUCKET/sac_tensorboard" --only-show-errors

echo "$(date -u +%FT%TZ) sync complete"
