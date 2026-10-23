#!/usr/bin/env bash
# Run LOCALLY if SSH starts failing after a network change — re-authorizes
# the security group for your current public IP (replaces the old rule).
set -euo pipefail
cd "$(dirname "$0")/.."
SG=$(cat aws/sg_id.txt)
OLD_IP=$(cat aws/my_ip.txt 2>/dev/null || true)
NEW_IP=$(curl -s https://checkip.amazonaws.com)

if [ -n "${OLD_IP:-}" ]; then
  aws ec2 revoke-security-group-ingress --profile carla-rl --region us-east-1 \
    --group-id "$SG" --protocol tcp --port 22 --cidr "${OLD_IP}/32" 2>/dev/null || true
fi
aws ec2 authorize-security-group-ingress --profile carla-rl --region us-east-1 \
  --group-id "$SG" --protocol tcp --port 22 --cidr "${NEW_IP}/32"
echo "$NEW_IP" > aws/my_ip.txt
echo "SSH now allowed from $NEW_IP"
