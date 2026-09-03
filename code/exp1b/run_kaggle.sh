#!/usr/bin/env bash
set -euo pipefail

PRESET="${1:-full}"
MODE="${2:-run}"
SHARD_INDEX="${3:-0}"
NUM_SHARDS="${4:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_PY="${SCRIPT_DIR}/run.py"
REQ_FILE="${CODE_ROOT}/requirements.txt"
OUTPUT="${KAGGLE_OUTPUT:-/kaggle/working/exp1b/data/${PRESET}}"

if [[ ! -f "$RUN_PY" ]]; then
  echo "error: missing ${RUN_PY}" >&2
  exit 1
fi

if [[ -f "$REQ_FILE" ]]; then
  python -m pip install -q -r "$REQ_FILE"
else
  echo "warning: ${REQ_FILE} not found; installing minimal atlas dependencies" >&2
  python -m pip install -q numpy scipy scikit-learn matplotlib
fi

cd "$CODE_ROOT"

NUM_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

run_python() {
  python "$RUN_PY" "$@"
}

if [[ "$NUM_GPUS" -gt 1 && "$MODE" != "analyze" ]]; then
  torchrun --standalone --nproc_per_node="$NUM_GPUS" "$RUN_PY" \
    --preset "$PRESET" \
    --mode "$MODE" \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    --distributed \
    --output "$OUTPUT"
else
  run_python \
    --preset "$PRESET" \
    --mode "$MODE" \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    --output "$OUTPUT"
fi

if [[ "$MODE" == "all" || "$MODE" == "analyze" ]]; then
  run_python \
    --preset "$PRESET" \
    --mode analyze \
    --output "$OUTPUT" \
    --bootstrap-draws "${BOOTSTRAP_DRAWS:-10000}"
fi
