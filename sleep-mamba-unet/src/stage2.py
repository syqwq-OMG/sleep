from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .postprocess import find_candidates, load_config


def _model():
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=31, random_state=42)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, random_state=42)


def build_candidate_features(pred_df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in candidates.itertuples(index=False):
        g = pred_df[pred_df["series_id"] == c.series_id].sort_values("step").reset_index(drop=True)
        idx = int(np.argmin(np.abs(g["step"].to_numpy() - c.step)))
        event_col = f"p_{c.event}"
        row = {"event_is_wakeup": float(c.event == "wakeup"), "peak_score": float(c.score)}
        for minutes in [1, 3, 5, 10, 30]:
            w = minutes * 12
            s = g[event_col].iloc[max(0, idx - w) : min(len(g), idx + w + 1)]
            row[f"mean_{minutes}m"] = float(s.mean())
            row[f"max_{minutes}m"] = float(s.max())
            row[f"sum_{minutes}m"] = float(s.sum())
        row["p_sleep_before"] = float(g["p_sleep"].iloc[max(0, idx - 120) : idx].mean()) if idx > 0 else 0.0
        row["p_sleep_after"] = float(g["p_sleep"].iloc[idx : min(len(g), idx + 120)].mean())
        row["sleep_contrast"] = row["p_sleep_after"] - row["p_sleep_before"]
        row["invalid_ratio"] = float(g.get("p_invalid", pd.Series(0, index=g.index)).iloc[max(0, idx - 120) : min(len(g), idx + 120)].mean())
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def label_candidates(candidates: pd.DataFrame, events: pd.DataFrame, tolerance_steps: int = 120) -> np.ndarray:
    y = []
    clean = events.dropna(subset=["series_id", "event", "step"])
    for c in candidates.itertuples(index=False):
        m = clean[
            (clean["series_id"] == c.series_id)
            & (clean["event"] == c.event)
            & ((clean["step"].astype(float) - float(c.step)).abs() <= tolerance_steps)
        ]
        y.append(float(len(m) > 0))
    return np.asarray(y, dtype=int)


def train_stage2(oof_pred: str, events_csv: str, out_dir: str, config: dict):
    pred = pd.read_parquet(oof_pred)
    events = pd.read_csv(events_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for event in ["onset", "wakeup"]:
        cand = find_candidates(pred, event, config)
        x = build_candidate_features(pred, cand)
        y = label_candidates(cand, events, tolerance_steps=int(config.get("stage2", {}).get("label_tolerance_steps", 120)))
        model = _model()
        if len(np.unique(y)) < 2:
            continue
        model.fit(x, y)
        joblib.dump({"model": model, "columns": list(x.columns)}, out / f"stage2_{event}.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--out-dir", default="outputs/stage2")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train_stage2(args.oof, args.events, args.out_dir, load_config(args.config))


if __name__ == "__main__":
    main()
