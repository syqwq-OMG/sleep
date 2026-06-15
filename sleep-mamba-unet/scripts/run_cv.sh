#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/default.yaml}
for FOLD in 0 1 2 3 4; do
  python -m src.train --config "$CONFIG" --fold "$FOLD"
done
