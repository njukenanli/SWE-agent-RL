#!/usr/bin/env bash
set -euo pipefail

EPOCH=""
DATA_DIR=""
INITIAL_MODEL="Qwen/Qwen3.5-4B"

usage() {
  cat <<'USAGE'
Usage:
  bash main.sh --epoch N --data-dir PATH

Options:
  --epoch N          Number of sequential vLLM + DAPO epochs to run.
  --data-dir PATH    Root directory for model/epoch_N HF models and checkpoints.
  -h, --help         Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epoch)
      EPOCH="${2:?missing value for --epoch}"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="${2:?missing value for --data-dir}"
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

if [[ -z "$DATA_DIR" ]]; then
  echo "Missing required argument: --data-dir" >&2
  usage >&2
  exit 2
fi

if ! [[ "$EPOCH" =~ ^[0-9]+$ ]] || [[ "$EPOCH" -lt 1 ]]; then
  echo "--epoch must be a positive integer, got: $EPOCH" >&2
  exit 2
fi

mkdir -p "$DATA_DIR/model" "$DATA_DIR/checkpoints"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

validate_hf_model() {
  local model_dir="$1"
  local weights=()

  if [[ ! -f "$model_dir/config.json" ]]; then
    echo "Missing Hugging Face config: $model_dir/config.json" >&2
    return 1
  fi
  if [[ ! -f "$model_dir/tokenizer_config.json" ]]; then
    echo "Missing Hugging Face tokenizer config: $model_dir/tokenizer_config.json" >&2
    return 1
  fi

  shopt -s nullglob
  weights=(
    "$model_dir"/*.safetensors
    "$model_dir"/pytorch_model*.bin
  )
  shopt -u nullglob

  if [[ "${#weights[@]}" -eq 0 ]]; then
    echo "No Hugging Face model weights found in: $model_dir" >&2
    return 1
  fi
}

MODEL_PATH="$INITIAL_MODEL"

for ((epoch = 0; epoch < EPOCH; epoch++)); do
  CHECKPOINT_PATH="$DATA_DIR/checkpoints/epoch_$epoch"
  HF_EXPORT_PATH="$CHECKPOINT_PATH/global_step_1/model/huggingface"
  MODEL_SAVE_PATH="$DATA_DIR/model/epoch_$epoch"

  if [[ -e "$CHECKPOINT_PATH" || -L "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint path already exists: $CHECKPOINT_PATH" >&2
    exit 1
  fi
  if [[ -e "$MODEL_SAVE_PATH" || -L "$MODEL_SAVE_PATH" ]]; then
    echo "Model path already exists: $MODEL_SAVE_PATH" >&2
    exit 1
  fi

  echo "[epoch $epoch/$((EPOCH - 1))] Starting vLLM server with model: $MODEL_PATH"
  python vllm/server.py --model "$MODEL_PATH"

  echo "[epoch $epoch/$((EPOCH - 1))] Starting DAPO: $MODEL_PATH -> $MODEL_SAVE_PATH"
  (
    cd verl
    python examples/sft/rft/dapo.py \
      --dapo-json dapo.json \
      --train-parquet dapo.parquet \
      --model-path "$MODEL_PATH" \
      --save-path "$CHECKPOINT_PATH" \
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

  validate_hf_model "$HF_EXPORT_PATH"
  ln -s "../checkpoints/epoch_$epoch/global_step_1/model/huggingface" "$MODEL_SAVE_PATH"
  validate_hf_model "$MODEL_SAVE_PATH"

  MODEL_PATH="$MODEL_SAVE_PATH"
done
