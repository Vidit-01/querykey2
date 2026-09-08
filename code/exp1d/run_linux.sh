#!/usr/bin/env bash
set -euo pipefail
PRESET="${1:-quick}"
SHARD_INDEX="${2:-0}"
NUM_SHARDS="${3:-1}"
DATA_FILE="${4:-}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-8}"
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

if [[ "$NUM_GPUS" -ge 1 ]]; then
  NPROC=$((NUM_GPUS * WORKERS_PER_GPU))
  echo "using ${NUM_GPUS} GPUs x ${WORKERS_PER_GPU} workers via torchrun (${NPROC} processes)"
  torchrun --standalone --nproc_per_node="$NPROC" code/exp1d/run.py \
    --preset "$PRESET" \
    --mode run \
    --shard-index "$SHARD_INDEX" \
    --num-shards "$NUM_SHARDS" \
    --workers-per-gpu "$WORKERS_PER_GPU" \
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
