#!/usr/bin/env bash
set -euo pipefail

PRESET="${1:-full}"
MODE="${2:-run}"
SHARD_INDEX="${3:-0}"
NUM_SHARDS="${4:-1}"
OUTPUT="${KAGGLE_OUTPUT:-/kaggle/working/exp1b/data/${PRESET}}"

cd "${KAGGLE_WORKING_DIR:-/kaggle/working}"
REPO_ROOT="${REPO_ROOT:-/kaggle/working}"
cd "$REPO_ROOT"

python -m pip install -q -r code/requirements.txt

NUM_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

if [[ "$NUM_GPUS" -gt 1 && "$MODE" != "analyze" ]]; then
  torchrun --standalone --nproc_per_node="$NUM_GPUS" code/exp1b/run.py \
    --preset "$PRESET" \
    --mode "$MODE" \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    --distributed \
    --output "$OUTPUT"
else
  python code/exp1b/run.py \
    --preset "$PRESET" \
    --mode "$MODE" \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    --output "$OUTPUT"
fi

if [[ "$MODE" == "all" || "$MODE" == "analyze" ]]; then
  python code/exp1b/run.py \
    --preset "$PRESET" \
    --mode analyze \
    --output "$OUTPUT" \
    --bootstrap-draws "${BOOTSTRAP_DRAWS:-10000}"
fi
