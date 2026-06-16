from __future__ import annotations

import argparse

import pandas as pd

from .metric import score_events
from .postprocess import load_config, predictions_to_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", nargs="+", required=True)
    parser.add_argument("--events", default="/kaggle/input/competitions/child-mind-institute-detect-sleep-states/train_events.csv")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred = pd.concat([pd.read_parquet(path) for path in args.pred], ignore_index=True)
    events = pd.read_csv(args.events).dropna(subset=["step"])
    events_val = events[events["series_id"].isin(pred["series_id"].unique())]
    sub = predictions_to_submission(pred, cfg)
    score = score_events(events_val, sub, cfg["postprocess"]["tolerances_steps"])
    print(f"OOF local score: {score}")
    print(sub.shape)
    print(sub["event"].value_counts())
    print(sub["score"].describe())


if __name__ == "__main__":
    main()
