from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

from .metric import score_events
from .postprocess import load_config, predictions_to_submission


def _parse_list(text: str, cast):
    return [cast(x) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="/kaggle/working/outputs/oof_pred.parquet")
    parser.add_argument("--events", default="/kaggle/input/competitions/child-mind-institute-detect-sleep-states/train_events.csv")
    parser.add_argument("--config", default="configs/kaggle_medium.yaml")
    parser.add_argument("--out-config", default="/kaggle/working/tuned_submit.yaml")
    parser.add_argument("--smooth", default="24,36")
    parser.add_argument("--distance", default="180,240")
    parser.add_argument("--threshold", default="0.0001")
    parser.add_argument("--max-events", default="100,200")
    args = parser.parse_args()

    base = load_config(args.config)
    pred = pd.read_parquet(args.pred)
    events = pd.read_csv(args.events).dropna(subset=["step"])
    events_val = events[events["series_id"].isin(pred["series_id"].unique())]

    best = None
    rows = []
    for smooth in _parse_list(args.smooth, int):
        for distance in _parse_list(args.distance, int):
            for threshold in _parse_list(args.threshold, float):
                for max_events in _parse_list(args.max_events, int):
                    cfg = copy.deepcopy(base)
                    pp = cfg.setdefault("postprocess", {})
                    pp["smooth_window_steps"] = smooth
                    pp["peak_distance_steps"] = distance
                    pp["score_threshold"] = threshold
                    pp["max_events_per_series_event"] = max_events
                    sub = predictions_to_submission(pred, cfg)
                    score = score_events(events_val, sub, pp["tolerances_steps"])
                    row = {
                        "score": score,
                        "smooth_window_steps": smooth,
                        "peak_distance_steps": distance,
                        "score_threshold": threshold,
                        "max_events_per_series_event": max_events,
                        "rows": len(sub),
                    }
                    rows.append(row)
                    print(row, flush=True)
                    if best is None or score > best["score"]:
                        best = row

    assert best is not None
    tuned = copy.deepcopy(base)
    tuned["postprocess"]["smooth_window_steps"] = best["smooth_window_steps"]
    tuned["postprocess"]["peak_distance_steps"] = best["peak_distance_steps"]
    tuned["postprocess"]["score_threshold"] = best["score_threshold"]
    tuned["postprocess"]["max_events_per_series_event"] = best["max_events_per_series_event"]
    out = Path(args.out_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(tuned, sort_keys=False))
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(out.with_suffix(".csv"), index=False)
    print(f"BEST {best}", flush=True)
    print(f"WROTE {out}", flush=True)


if __name__ == "__main__":
    main()
