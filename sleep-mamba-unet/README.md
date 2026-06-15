# Sleep-Mamba-UNet for Detect Sleep States

Kaggle-ready baseline and extensible Sleep-Mamba-UNet pipeline for the Child Mind Institute Detect Sleep States competition.

## Quick Start

```bash
pip install -r requirements.txt
bash scripts/download_data.sh
python -m src.train --config configs/debug.yaml --fold 0
python -m src.infer --config configs/debug.yaml --checkpoints outputs/models/fold0.pt --phase test
python -m src.postprocess --pred outputs/test_pred.parquet --out outputs/submissions/submission.csv
pytest tests
```

Expected Kaggle submission columns:

```text
row_id,series_id,step,event,score
```

## Data Layout

Place Kaggle files under `data/raw/`:

```text
data/raw/train_series.parquet
data/raw/train_events.csv
data/raw/test_series.parquet
```

Download requires accepted competition rules and a configured Kaggle API token:

```bash
bash scripts/download_data.sh
```

## Training

Debug:

```bash
python -m src.train --config configs/debug.yaml --fold 0
```

Full fold:

```bash
python -m src.train --config configs/default.yaml --fold 0
```

All folds:

```bash
bash scripts/run_cv.sh
```

Splits are always grouped by `series_id`.

## Inference and Submission

```bash
python -m src.infer --config configs/default.yaml --checkpoints outputs/models/fold0.pt --phase test
python -m src.postprocess --pred outputs/test_pred.parquet --out outputs/submissions/submission.csv
bash scripts/make_submission.sh outputs/submissions/submission.csv "Sleep-Mamba-UNet v1"
```

Inference uses sliding windows and overlap averaging. Postprocessing turns per-step event probabilities into peak candidates and writes a valid submission even for empty/debug predictions.

## Kaggle Notebook Workflow

Upload this project as a Kaggle Dataset, attach it and the competition data to a Notebook, then run one cell. Change only `PRESET` while iterating.

```python
import os, shutil, sys

SRC_CODE_DIR = "/kaggle/input/datasets/syqwqomg/new-mamba-code/sleep-mamba-unet"
WORK_CODE_DIR = "/kaggle/working/sleep-mamba-unet"

if os.path.exists(WORK_CODE_DIR):
    shutil.rmtree(WORK_CODE_DIR)
shutil.copytree(SRC_CODE_DIR, WORK_CODE_DIR)

os.chdir(WORK_CODE_DIR)
sys.path.insert(0, WORK_CODE_DIR)

PRESET = "small"  # tiny, small, medium, large
FOLD = 0

train_cfg = f"configs/kaggle_{PRESET}.yaml"
submit_cfg = f"configs/kaggle_{PRESET}_submit.yaml"
ckpt = f"/kaggle/working/outputs/models/fold{FOLD}.pt"

!python -m src.train --config {train_cfg} --fold {FOLD}

import pandas as pd
from src.postprocess import load_config, predictions_to_submission
from src.metric import score_events

cfg = load_config(train_cfg)
pred = pd.read_parquet("/kaggle/working/outputs/oof_pred.parquet")
events = pd.read_csv("/kaggle/input/competitions/child-mind-institute-detect-sleep-states/train_events.csv")
events = events.dropna(subset=["step"])
events_val = events[events["series_id"].isin(pred["series_id"].unique())]
val_sub = predictions_to_submission(pred, cfg)
print("local score:", score_events(events_val, val_sub, cfg["postprocess"]["tolerances_steps"]))
print(val_sub.head())
print(val_sub.shape)

!python -m src.infer --config {submit_cfg} --checkpoints {ckpt} --phase test
!python -m src.postprocess --pred /kaggle/working/outputs/test_pred.parquet --out /kaggle/working/submission.csv --config {submit_cfg}

sub = pd.read_csv("/kaggle/working/submission.csv")
print(sub.head())
print(sub.shape)
print(sub["event"].value_counts())
print(sub["score"].describe())
```

Available presets:

```text
tiny    fast smoke test, lowest memory
small   known runnable baseline
medium  more series and epochs, try after small
large   more series, higher memory risk
```

The train configs use `debug.n_series` caps to fit Kaggle RAM. The submit configs do not include `debug`, so test inference is not truncated.

## Notes

- No `mamba-ssm` dependency is required. `MambaLiteBlock` is pure PyTorch.
- Stage 2 candidate calibration is optional and only trains from OOF predictions.
- The local EDAP metric implements the competition-style event average precision over onset/wakeup and configured tolerances.
