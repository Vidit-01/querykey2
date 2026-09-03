#!/usr/bin/env bash
set -euo pipefail
PRESET="${1:-quick}"
STUDY="${2:-pilot}"
SHARD_INDEX="${3:-0}"
NUM_SHARDS="${4:-1}"
TRAIN_FILE="${5:-}"
VALIDATION_FILE="${6:-}"
EXTRA=()
if [[ -n "$TRAIN_FILE" ]]; then
  EXTRA+=(--train-file "$TRAIN_FILE")
fi
if [[ -n "$VALIDATION_FILE" ]]; then
  EXTRA+=(--validation-file "$VALIDATION_FILE")
fi
if [[ ! -f "code/exp2/data/$PRESET/frozen_treatments.json" ]]; then
  python code/exp2/run.py --preset "$PRESET" --study "$STUDY" --mode freeze
fi
if [[ ! -f "code/exp2/data/$PRESET/integration_check.json" ]]; then
  python code/exp2/run.py --preset "$PRESET" --study "$STUDY" --mode integration
fi
python code/exp2/run.py \
  --preset "$PRESET" \
  --study "$STUDY" \
  --mode run \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  "${EXTRA[@]}"
if [[ "$NUM_SHARDS" == "1" ]]; then
  python code/exp2/run.py --preset "$PRESET" --study "$STUDY" --mode analyze
fi
