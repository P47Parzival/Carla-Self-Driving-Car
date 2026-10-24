#!/usr/bin/env bash
# Run LOCALLY. Starts the stopped instance back up and prints its new public IP
# (EC2 public IPs change on every stop/start unless you allocate an Elastic IP).
set -euo pipefail
cd "$(dirname "$0")/.."
REGION=$(cat aws/region.txt)
ID=$(cat aws/instance_id.txt)
aws ec2 start-instances --profile carla-rl --region "$REGION" --instance-ids "$ID"
echo "Starting $ID, waiting for it to be running..."
aws ec2 wait instance-running --profile carla-rl --region "$REGION" --instance-ids "$ID"
IP=$(aws ec2 describe-instances --profile carla-rl --region "$REGION" --instance-ids "$ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$IP" > aws/instance_ip.txt
echo "Running at $IP"
