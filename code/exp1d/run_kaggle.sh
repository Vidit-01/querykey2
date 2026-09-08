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

python "$RUN_PY" \
  --preset "$PRESET" \
  --mode "$MODE" \
  --output "$OUTPUT" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  "${EXTRA[@]}"
