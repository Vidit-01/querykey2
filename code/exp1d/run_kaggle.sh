#!/usr/bin/env bash
set -euo pipefail

PRESET="${1:-full}"
MODE="${2:-run}"
SHARD_INDEX="${3:-0}"
NUM_SHARDS="${4:-1}"
DATA_FILE="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_PY="${SCRIPT_DIR}/run.py"
REQ_FILE="${CODE_ROOT}/requirements.txt"
OUTPUT="${KAGGLE_OUTPUT:-/kaggle/working/exp1d/data/${PRESET}}"

if [[ ! -f "$RUN_PY" ]]; then
  echo "error: missing ${RUN_PY}" >&2
  exit 1
fi

if [[ -f "$REQ_FILE" ]]; then
  python -m pip install -q -r "$REQ_FILE"
fi

EXTRA=()
if [[ -n "$DATA_FILE" ]]; then
  EXTRA+=(--tinystories-file "$DATA_FILE")
elif [[ -n "${KAGGLE_TINYSTORIES:-}" && -f "${KAGGLE_TINYSTORIES}" ]]; then
  EXTRA+=(--tinystories-file "$KAGGLE_TINYSTORIES")
fi

echo "output: ${OUTPUT}"
cd "$CODE_ROOT"

COMMON_ARGS=(
  --preset "$PRESET"
  --output "$OUTPUT"
  "${EXTRA[@]}"
)

NUM_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
echo "visible GPUs: ${NUM_GPUS}"

run_python() {
  python "$RUN_PY" "$@"
}

if [[ "$MODE" == "select" || "$MODE" == "all" ]]; then
  run_python --mode select "${COMMON_ARGS[@]}"
fi

if [[ "$MODE" == "run" || "$MODE" == "all" ]]; then
  if [[ "$NUM_GPUS" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NUM_GPUS" "$RUN_PY" \
      --mode run \
      "${COMMON_ARGS[@]}" \
      --shard-index "$SHARD_INDEX" \
      --num-shards "$NUM_SHARDS" \
      --distributed
  else
    run_python \
      --mode run \
      "${COMMON_ARGS[@]}" \
      --shard-index "$SHARD_INDEX" \
      --num-shards "$NUM_SHARDS"
  fi
fi

if [[ "$MODE" == "analyze" || "$MODE" == "all" ]]; then
  run_python \
    --mode analyze \
    "${COMMON_ARGS[@]}" \
    --bootstrap-draws "${BOOTSTRAP_DRAWS:-10000}"
fi
