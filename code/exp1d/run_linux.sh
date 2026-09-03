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
python code/exp1d/run.py \
  --preset "$PRESET" \
  --mode run \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  "${EXTRA[@]}"
if [[ "$NUM_SHARDS" == "1" ]]; then
  python code/exp1d/run.py --preset "$PRESET" --mode analyze
fi
