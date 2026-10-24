#!/usr/bin/env bash
# Run LOCALLY when you're fully done. Terminates the instance and deletes the
# security group + key pair. Does NOT delete the S3 bucket (your checkpoints) —
# remove that yourself if you no longer want it.
set -euo pipefail
cd "$(dirname "$0")/.."
REGION=$(cat aws/region.txt)

ID=$(cat aws/instance_id.txt 2>/dev/null || true)
SG=$(cat aws/sg_id.txt 2>/dev/null || true)

if [ -n "${ID:-}" ]; then
  aws ec2 terminate-instances --profile carla-rl --region "$REGION" --instance-ids "$ID"
  aws ec2 wait instance-terminated --profile carla-rl --region "$REGION" --instance-ids "$ID"
  echo "Terminated $ID"
fi

if [ -n "${SG:-}" ]; then
  aws ec2 delete-security-group --profile carla-rl --region "$REGION" --group-id "$SG"
  echo "Deleted security group $SG"
fi

aws ec2 delete-key-pair --profile carla-rl --region "$REGION" --key-name carla-rl-key
echo "Deleted key pair. Checkpoints remain in s3://$(cat aws/bucket_name.txt 2>/dev/null)."
