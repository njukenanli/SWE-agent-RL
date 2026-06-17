#!/usr/bin/env bash
set -euo pipefail

DEFAULT_MODEL="Qwen/Qwen3.5-4B"

MODEL="$DEFAULT_MODEL"
HOST="0.0.0.0"
PORT="8000"
SERVED_MODEL_NAME=""
API_KEY=""
MAX_LOGPROBS="1"
LOGPROBS_MODE="processed_logprobs"
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage:
  bash vllm.sh [--model MODEL_OR_PATH] [options] [-- extra vLLM args]

Options:
  --model MODEL_OR_PATH       Hugging Face model name or local model path.
                              Default: Qwen/Qwen3.5-4B
  --host HOST                 Server bind host. Default: 0.0.0.0
  --port PORT                 Server port. Default: 8000
  --served-model-name NAME    API model name. Default: same as --model
  --api-key KEY               Optional OpenAI-compatible API key.
  --max-logprobs N            Max logprobs the server will return. Default: 1
  --logprobs-mode MODE        processed_logprobs, raw_logprobs, processed_logits,
                              or raw_logits. Default: processed_logprobs
  --dry-run                   Print the vLLM command without running it.
  -h, --help                  Show this help.

For token IDs and token probabilities, send these request fields to the
OpenAI-compatible endpoint:
  return_token_ids: true
  logprobs: true
  top_logprobs: 1
  prompt_logprobs: 1

The server defaults below keep sampling neutral except for temperature=1.0:
  temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:?missing value for --model}"
      shift 2
      ;;
    --host)
      HOST="${2:?missing value for --host}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="${2:?missing value for --served-model-name}"
      shift 2
      ;;
    --api-key)
      API_KEY="${2:?missing value for --api-key}"
      shift 2
      ;;
    --max-logprobs)
      MAX_LOGPROBS="${2:?missing value for --max-logprobs}"
      shift 2
      ;;
    --logprobs-mode)
      LOGPROBS_MODE="${2:?missing value for --logprobs-mode}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SERVED_MODEL_NAME" ]]; then
  SERVED_MODEL_NAME="$MODEL"
fi

GENERATION_CONFIG='{"temperature":1.0,"top_p":1.0,"top_k":0,"min_p":0.0,"repetition_penalty":1.0}'

if [[ -n "${VLLM_BIN:-}" ]]; then
  # Allows: VLLM_BIN="python -m vllm.entrypoints.openai.api_server" bash vllm.sh
  read -r -a VLLM_CMD <<< "$VLLM_BIN"
elif command -v vllm >/dev/null 2>&1; then
  VLLM_CMD=(vllm serve)
else
  VLLM_CMD=(python -m vllm.entrypoints.openai.api_server)
fi

CMD=(
  "${VLLM_CMD[@]}"
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --generation-config vllm
  --override-generation-config "$GENERATION_CONFIG"
  --max-logprobs "$MAX_LOGPROBS"
  --logprobs-mode "$LOGPROBS_MODE"
)

if [[ -n "$API_KEY" ]]; then
  CMD+=(--api-key "$API_KEY")
fi

CMD+=("${EXTRA_ARGS[@]}")

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

exec "${CMD[@]}"
