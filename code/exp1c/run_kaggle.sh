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
OUTPUT="${KAGGLE_OUTPUT:-/kaggle/working/exp1c/data/${PRESET}}"

if [[ ! -f "$RUN_PY" ]]; then
  echo "error: missing ${RUN_PY}" >&2
  exit 1
fi

if [[ -f "$REQ_FILE" ]]; then
  python -m pip install -q -r "$REQ_FILE"
else
  echo "warning: ${REQ_FILE} not found; installing minimal dependencies" >&2
  python -m pip install -q numpy scipy scikit-learn matplotlib joblib pandas
fi

resolve_discovery_summary() {
  if [[ -n "${KAGGLE_DISCOVERY_SUMMARY:-}" && -f "${KAGGLE_DISCOVERY_SUMMARY}" ]]; then
    echo "${KAGGLE_DISCOVERY_SUMMARY}"
    return
  fi
  local candidate
  for candidate in \
    "/kaggle/working/exp1b/data/${PRESET}/atlas_summary.csv" \
    "${CODE_ROOT}/exp1b/data/${PRESET}/atlas_summary.csv"; do
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
  if [[ -d /kaggle/input ]]; then
    candidate="$(find /kaggle/input -name atlas_summary.csv -print -quit)"
    if [[ -n "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  fi
  echo ""
}

DISCOVERY_SUMMARY="$(resolve_discovery_summary)"
if [[ -z "$DISCOVERY_SUMMARY" ]]; then
  echo "error: atlas_summary.csv not found." >&2
  echo "Attach Experiment 1B summary as a Kaggle dataset or set KAGGLE_DISCOVERY_SUMMARY." >&2
  exit 1
fi

echo "discovery summary: ${DISCOVERY_SUMMARY}"
echo "output: ${OUTPUT}"

cd "$CODE_ROOT"

COMMON_ARGS=(
  --preset "$PRESET"
  --output "$OUTPUT"
  --discovery-summary "$DISCOVERY_SUMMARY"
)

NUM_GPUS="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"

run_python() {
  python "$RUN_PY" "$@"
}

if [[ "$MODE" == "propose" || "$MODE" == "all" ]]; then
  run_python --mode propose "${COMMON_ARGS[@]}"
fi

if [[ "$MODE" == "run" || "$MODE" == "all" ]]; then
  if [[ ! -f "${OUTPUT}/adaptive_points.json" ]]; then
    echo "adaptive_points.json missing; running propose first..."
    run_python --mode propose "${COMMON_ARGS[@]}"
  fi
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
