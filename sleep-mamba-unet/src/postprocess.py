from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


EVENTS = ("onset", "wakeup")


def load_config(path: str | Path | None) -> dict:
    if path is None:
        return {}
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base_config", None)
    if base:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent.parent / base if not (path.parent / base).exists() else path.parent / base
        base_cfg = load_config(base_path)
        return _merge(base_cfg, cfg)
    return cfg


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def validate_submission(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["row_id", "series_id", "step", "event", "score"]
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"submission missing {col}")
    out = df[cols].copy()
    out["row_id"] = np.arange(len(out), dtype=int)
    out["step"] = out["step"].astype(int)
    out["score"] = out["score"].astype(float).clip(0, 1)
    bad = set(out["event"]) - set(EVENTS)
    if bad:
        raise ValueError(f"invalid events: {bad}")
    return out


def find_candidates(pred_df: pd.DataFrame, event: str, config: dict | None = None) -> pd.DataFrame:
    if event not in EVENTS:
        raise ValueError(event)
    cfg = (config or {}).get("postprocess", config or {})
    smooth = int(cfg.get("smooth_window_steps", 12))
    distance = int(cfg.get("peak_distance_steps", 180))
    threshold = float(cfg.get("score_threshold", 0.001))
    max_n = int(cfg.get("max_events_per_series_event", 200))
    prob_col = f"p_{event}"
    rows = []
    for series_id, g in pred_df.groupby("series_id", sort=False):
        g = g.sort_values("step").reset_index(drop=True)
        if prob_col not in g:
            continue
        values = g[prob_col].fillna(0).to_numpy(dtype=float)
        smooth_values = uniform_filter1d(values, size=max(smooth, 1), mode="nearest") if len(values) else values
        peaks, props = find_peaks(smooth_values, height=threshold, distance=max(distance, 1))
        if len(peaks) == 0 and len(g) > 0:
            peaks = np.array([int(np.argmax(smooth_values))])
            props = {"peak_heights": smooth_values[peaks]}
        order = np.argsort(props.get("peak_heights", smooth_values[peaks]))[::-1][:max_n]
        for rank in order:
            idx = int(peaks[rank])
            peak_score = float(smooth_values[idx])
            mass = float(values[max(0, idx - 12) : min(len(values), idx + 13)].mean()) if len(values) else 0.0
            invalid = float(g.get("p_invalid", pd.Series(0, index=g.index)).iloc[max(0, idx - 12) : min(len(g), idx + 13)].mean())
            score = np.clip(0.85 * peak_score + 0.15 * mass - 0.05 * invalid, 0, 1)
            rows.append({"series_id": series_id, "step": int(g.loc[idx, "step"]), "event": event, "score": float(score)})
    return pd.DataFrame(rows, columns=["series_id", "step", "event", "score"])


def predictions_to_submission(pred_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    parts = [find_candidates(pred_df, event, config) for event in EVENTS]
    sub = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if sub.empty:
        series_ids = pred_df["series_id"].drop_duplicates().tolist() if "series_id" in pred_df else ["dummy"]
        sub = pd.DataFrame(
            [{"series_id": sid, "step": 0, "event": event, "score": 0.0} for sid in series_ids for event in EVENTS]
        )
    sub = sub.sort_values(["series_id", "event", "score"], ascending=[True, True, False]).reset_index(drop=True)
    sub.insert(0, "row_id", np.arange(len(sub), dtype=int))
    return validate_submission(sub)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else {}
    pred = pd.read_parquet(args.pred)
    sub = predictions_to_submission(pred, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"wrote {out} rows={len(sub)}")


if __name__ == "__main__":
    main()
