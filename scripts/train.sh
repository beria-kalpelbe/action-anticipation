#!/usr/bin/env bash
# Launch a config-defined action-anticipation experiment from any directory.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 CONFIG TRAIN_CSV VAL_CSV FEATURE_STORE OUTPUT_DIR" >&2
  exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

exec uv run --directory "${repo_root}" aa-train \
  --config "$1" \
  --train-csv "$2" \
  --val-csv "$3" \
  --feature-store "$4" \
  --output "$5"
