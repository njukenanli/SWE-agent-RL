#!/usr/bin/env bash
set -euo pipefail

EPOCH=""
VLLM_MODEL="Qwen/Qwen3.5-4B"

usage() {
  cat <<'USAGE'
Usage:
  bash main.sh --epoch N

Options:
  --epoch N     Number of sequential vLLM + DAPO rounds to run.
  -h, --help    Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epoch)
      EPOCH="${2:?missing value for --epoch}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$EPOCH" ]]; then
  echo "Missing required argument: --epoch" >&2
  usage >&2
  exit 2
fi

if ! [[ "$EPOCH" =~ ^[0-9]+$ ]] || [[ "$EPOCH" -lt 1 ]]; then
  echo "--epoch must be a positive integer, got: $EPOCH" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for ((round = 1; round <= EPOCH; round++)); do
  echo "[$round/$EPOCH] Starting vLLM server"
  python vllm/server.py --model "$VLLM_MODEL"

  echo "[$round/$EPOCH] Starting DAPO"
  (
    cd verl
    python examples/sft/rft/dapo.py \
      --dapo-json dapo.json \
      --train-parquet dapo.parquet \
      --model-path Qwen/Qwen3-8B \
      --save-path checkpoints/dapo/qwen3-8b-megatron \
      --num-groups 64 \
      --nproc 8 \
      --tp 1 \
      --pp 1 \
      --cp 1 \
      --micro-batch-size-per-gpu 1 \
      --max-input-length 60000 \
      --max-total-length 60000 \
      --max-token-len-per-gpu 60000 \
      --max-steps 50 \
      --step-penalty-threshold 35 \
      --clip-ratio-low 0.2 \
      --clip-ratio-high 0.28 \
      --lr 1e-6 \
      --min-lr 1e-6
  )
done
