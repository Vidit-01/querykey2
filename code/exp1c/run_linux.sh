#!/usr/bin/env bash
set -euo pipefail
PRESET="${1:-quick}"
SHARD_INDEX="${2:-0}"
NUM_SHARDS="${3:-1}"
EXTRA=()
if [[ "$PRESET" == "full" ]]; then
  EXTRA+=(--replicate-discovery-grid)
fi
if [[ ! -f "code/exp1c/data/$PRESET/adaptive_points.json" ]]; then
  python code/exp1c/run.py --preset "$PRESET" --mode propose
fi
python code/exp1c/run.py \
  --preset "$PRESET" \
  --mode run \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  "${EXTRA[@]}"
if [[ "$NUM_SHARDS" == "1" ]]; then
  python code/exp1c/run.py --preset "$PRESET" --mode analyze
fi
