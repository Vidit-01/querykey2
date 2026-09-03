#!/usr/bin/env bash
set -euo pipefail
PRESET="${1:-quick}"
SHARD_INDEX="${2:-0}"
NUM_SHARDS="${3:-1}"
python code/exp1a/run.py \
  --preset "$PRESET" \
  --mode run \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS"
if [[ "$NUM_SHARDS" == "1" ]]; then
  python code/exp1a/run.py --preset "$PRESET" --mode analyze
fi
