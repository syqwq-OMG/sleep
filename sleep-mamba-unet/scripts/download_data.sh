#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
kaggle competitions download -c child-mind-institute-detect-sleep-states -p data/raw
unzip -o data/raw/child-mind-institute-detect-sleep-states.zip -d data/raw
