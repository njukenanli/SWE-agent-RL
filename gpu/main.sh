#!/usr/bin/env bash
set -euo pipefail

EPOCH=""
START_EPOCH="0"
DATA_DIR=""
INITIAL_MODEL="Qwen/Qwen3.5-4B"
CONTEXT_PARALLEL="1"
VLLM_TENSOR_PARALLEL="auto"

usage() {
  cat <<'USAGE'
Usage:
  bash main.sh --epoch N --data-dir PATH [--start_epoch N] [options]

Options:
  --epoch N             Total target epoch count; the final epoch is N-1.
  --start_epoch N       First zero-based epoch to run. Default: 0.
  --data-dir PATH       Root directory for model/epoch_N HF models and checkpoints.
  --context-parallel N  Context-parallel GPU count for DAPO. Default: 1.
  --vllm-tensor-parallel N
                        Tensor-parallel GPU count for vLLM. Default: all visible GPUs.
  -h, --help            Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epoch)
      EPOCH="${2:?missing value for --epoch}"
      shift 2
      ;;
    --start_epoch)
      START_EPOCH="${2:?missing value for --start_epoch}"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="${2:?missing value for --data-dir}"
      shift 2
      ;;
    --context-parallel)
      CONTEXT_PARALLEL="${2:?missing value for --context-parallel}"
      shift 2
      ;;
    --vllm-tensor-parallel)
      VLLM_TENSOR_PARALLEL="${2:?missing value for --vllm-tensor-parallel}"
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

if ! [[ "$START_EPOCH" =~ ^[0-9]+$ ]]; then
  echo "--start_epoch must be a non-negative integer, got: $START_EPOCH" >&2
  exit 2
fi

if [[ "$START_EPOCH" -ge "$EPOCH" ]]; then
  echo "--start_epoch must be less than --epoch, got: $START_EPOCH >= $EPOCH" >&2
  exit 2
fi

if ! [[ "$CONTEXT_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$CONTEXT_PARALLEL" -lt 1 ]]; then
  echo "--context-parallel must be a positive integer, got: $CONTEXT_PARALLEL" >&2
  exit 2
fi

if [[ "$VLLM_TENSOR_PARALLEL" != "auto" ]] &&
  { ! [[ "$VLLM_TENSOR_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$VLLM_TENSOR_PARALLEL" -lt 1 ]]; }; then
  echo "--vllm-tensor-parallel must be a positive integer, got: $VLLM_TENSOR_PARALLEL" >&2
  exit 2
fi

mkdir -p "$DATA_DIR/model" "$DATA_DIR/checkpoints"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

detect_num_gpus() {
  local count

  if ! count="$(python -c 'import torch; print(torch.cuda.device_count())')"; then
    echo "Failed to detect visible CUDA GPUs with PyTorch" >&2
    return 1
  fi
  if ! [[ "$count" =~ ^[0-9]+$ ]] || [[ "$count" -lt 1 ]]; then
    echo "Expected at least one visible CUDA GPU, detected: $count" >&2
    return 1
  fi

  printf '%s\n' "$count"
}

NUM_GPUS="$(detect_num_gpus)"
if [[ "$VLLM_TENSOR_PARALLEL" == "auto" ]]; then
  VLLM_TENSOR_PARALLEL="$NUM_GPUS"
elif (( VLLM_TENSOR_PARALLEL > NUM_GPUS )); then
  echo "--vllm-tensor-parallel ($VLLM_TENSOR_PARALLEL) cannot exceed the visible GPU count ($NUM_GPUS)" >&2
  exit 2
fi
if (( NUM_GPUS % CONTEXT_PARALLEL != 0 )); then
  echo "Detected GPU count ($NUM_GPUS) must be divisible by --context-parallel ($CONTEXT_PARALLEL)" >&2
  exit 2
fi
echo "Detected $NUM_GPUS visible CUDA GPU(s); vLLM will use tensor parallelism $VLLM_TENSOR_PARALLEL, and DAPO will use $NUM_GPUS process(es) with context parallelism $CONTEXT_PARALLEL."

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

if [[ "$START_EPOCH" -eq 0 ]]; then
  MODEL_PATH="$INITIAL_MODEL"
else
  PREVIOUS_EPOCH=$((START_EPOCH - 1))
  MODEL_PATH="$DATA_DIR/model/epoch_$PREVIOUS_EPOCH"
  validate_hf_model "$MODEL_PATH"
  echo "Resuming at epoch $START_EPOCH with model from epoch $PREVIOUS_EPOCH: $MODEL_PATH"
fi

for ((epoch = START_EPOCH; epoch < EPOCH; epoch++)); do
  CHECKPOINT_PATH="$DATA_DIR/checkpoints/epoch_$epoch"
  HF_EXPORT_PATH="$CHECKPOINT_PATH/global_step_1/model/huggingface"
  MODEL_SAVE_PATH="$DATA_DIR/model/epoch_$epoch"

  if [[ -e "$CHECKPOINT_PATH" || -L "$CHECKPOINT_PATH" ]] &&
    [[ -e "$MODEL_SAVE_PATH" || -L "$MODEL_SAVE_PATH" ]] &&
    validate_hf_model "$MODEL_SAVE_PATH"; then
    echo "Model path already exists: $MODEL_SAVE_PATH. If you want to skip this epoch, check the model checkpoint under $MODEL_SAVE_PATH is complete and then increase --start_epoch argument; if you want to redo this epoch, manually delete model checkpoints at $CHECKPOINT_PATH and $MODEL_SAVE_PATH and all checkpoints of epochs after it." >&2
    exit 1
  fi

  if [[ -e "$CHECKPOINT_PATH" || -L "$CHECKPOINT_PATH" ]]; then
    echo "Removing incomplete checkpoint path: $CHECKPOINT_PATH" >&2
    rm -rf -- "$CHECKPOINT_PATH"
  fi
  if [[ -e "$MODEL_SAVE_PATH" || -L "$MODEL_SAVE_PATH" ]]; then
    echo "Removing incomplete model path: $MODEL_SAVE_PATH" >&2
    rm -rf -- "$MODEL_SAVE_PATH"
  fi

  echo "[epoch $epoch/$((EPOCH - 1))] Starting vLLM server with model: $MODEL_PATH"
  python vllm/server.py \
    --model "$MODEL_PATH" \
    --epoch "$epoch" \
    -- \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL"

  echo "[epoch $epoch/$((EPOCH - 1))] Starting DAPO: $MODEL_PATH -> $MODEL_SAVE_PATH"
  (
    cd verl
    python examples/sft/rft/dapo.py \
      --dapo-json dapo.json \
      --train-parquet dapo.parquet \
      --model-path "$MODEL_PATH" \
      --save-path "$CHECKPOINT_PATH" \
      --num-groups 0 \
      --nproc "$NUM_GPUS" \
      --tp 1 \
      --pp 1 \
      --cp "$CONTEXT_PARALLEL" \
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
