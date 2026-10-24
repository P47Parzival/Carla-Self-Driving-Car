#!/usr/bin/env bash
# Runs ON the EC2 instance once. Schedules sync_to_s3.sh every 15 minutes.
set -euo pipefail

SCRIPT="$HOME/carla-rl/project/aws/sync_to_s3.sh"
chmod +x "$SCRIPT"
( crontab -l 2>/dev/null | grep -v "$SCRIPT" || true ; echo "*/15 * * * * $SCRIPT >> $HOME/carla-rl/sync.log 2>&1" ) | crontab -
echo "Cron installed. Logs at ~/carla-rl/sync.log"
