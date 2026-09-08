#!/usr/bin/env bash
set -euo pipefail
PRESET="${1:-quick}"
SHARD_INDEX="${2:-0}"
NUM_SHARDS="${3:-1}"
DATA_FILE="${4:-}"
EXTRA=()
if [[ -n "$DATA_FILE" ]]; then
  EXTRA+=(--tinystories-file "$DATA_FILE")
fi
if [[ ! -f "code/exp1d/data/$PRESET/selected_points.json" ]]; then
  python code/exp1d/run.py --preset "$PRESET" --mode select
fi

NUM_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

if [[ "$NUM_GPUS" -gt 1 ]]; then
  echo "using ${NUM_GPUS} GPUs via torchrun"
  torchrun --standalone --nproc_per_node="$NUM_GPUS" code/exp1d/run.py \
    --preset "$PRESET" \
    --mode run \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    --distributed \
    "${EXTRA[@]}"
else
  python code/exp1d/run.py \
    --preset "$PRESET" \
    --mode run \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    "${EXTRA[@]}"
fi
if [[ "$NUM_SHARDS" == "1" ]]; then
  python code/exp1d/run.py --preset "$PRESET" --mode analyze
fi
