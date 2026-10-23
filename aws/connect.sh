#!/usr/bin/env bash
# Run LOCALLY. SSHes in and tunnels TensorBoard (6006) over the SSH
# connection instead of opening it to the public internet.
set -euo pipefail

cd "$(dirname "$0")/.."
IP=$(cat aws/instance_ip.txt)
KEY="aws/carla-rl-key.pem"

echo "TensorBoard will be reachable at http://localhost:6006 once started on the instance."
ssh -i "$KEY" -L 6006:localhost:6006 -o StrictHostKeyChecking=accept-new "ubuntu@$IP"
