#!/usr/bin/env bash
set -euo pipefail
FILE=${1:-outputs/submissions/submission.csv}
MSG=${2:-"Sleep-Mamba-UNet"}
kaggle competitions submit -c child-mind-institute-detect-sleep-states -f "$FILE" -m "$MSG"
