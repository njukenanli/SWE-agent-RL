#!/usr/bin/env bash

set -u

retry_delay_seconds="${SSH_TUNNEL_RETRY_DELAY_SECONDS:-10}"

trap 'echo "Stopping SSH tunnel."; exit 0' INT TERM

while true; do
  echo "Starting SSH tunnel..."

  ssh -i key.pem \
    -L 5001:localhost:5001 \
    -L 9001:localhost:9001 \
    -L 8004:localhost:8004 \
    -R 8003:localhost:8003 \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=60 \
    -N user@aws.com

  exit_code=$?
  echo "SSH tunnel exited with status ${exit_code}; retrying in ${retry_delay_seconds} seconds."
  sleep "${retry_delay_seconds}"
done
