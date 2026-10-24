#!/usr/bin/env bash
# Run LOCALLY. Stops the instance (keeps EBS disk, stops GPU/compute billing).
set -euo pipefail
cd "$(dirname "$0")/.."
REGION=$(cat aws/region.txt)
ID=$(cat aws/instance_id.txt)
aws ec2 stop-instances --profile carla-rl --region "$REGION" --instance-ids "$ID"
echo "Stopping $ID. EBS storage still bills (~\$0.01/GB-month); compute stops."
