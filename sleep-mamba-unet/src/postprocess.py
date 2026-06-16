from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


EVENTS = ("onset", "wakeup")
SUBMISSION_COLUMNS = ["row_id", "series_id", "step", "event", "score"]


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
    for col in SUBMISSION_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"submission missing {col}")
    out = df[SUBMISSION_COLUMNS].copy()
    out["row_id"] = np.arange(len(out), dtype=int)
    out["step"] = out["step"].astype(int)
    out["score"] = out["score"].astype(float).clip(0, 1)
    bad = set(out["event"]) - set(EVENTS)
    if bad:
        raise ValueError(f"invalid events: {bad}")
    return out


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=["series_id", "step", "event", "score"])


def find_candidates(pred_df: pd.DataFrame, event: str, config: dict | None = None) -> pd.DataFrame:
    if event not in EVENTS:
        raise ValueError(event)
    cfg = (config or {}).get("postprocess", config or {})
    smooth = int(cfg.get("smooth_window_steps", 12))
    distance = int(cfg.get("peak_distance_steps", 180))
    threshold = float(cfg.get("score_threshold", 0.001))
    max_n = int(cfg.get("max_events_per_series_event", 200))
    peak_weight = float(cfg.get("peak_weight", 0.85))
    mass_weight = float(cfg.get("mass_weight", 0.15))
    invalid_penalty = float(cfg.get("invalid_penalty", 0.0))
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
            score = np.clip(peak_weight * peak_score + mass_weight * mass - invalid_penalty * invalid, 0, 1)
            rows.append({"series_id": series_id, "step": int(g.loc[idx, "step"]), "event": event, "score": float(score)})
    return pd.DataFrame(rows, columns=["series_id", "step", "event", "score"])


def _fallback_rows(candidates: pd.DataFrame, used: set[tuple[str, str, int]], keep_per_event: int, score_scale: float) -> pd.DataFrame:
    if keep_per_event <= 0 or candidates.empty:
        return _empty_events()
    rows = []
    for (series_id, event), g in candidates.groupby(["series_id", "event"], sort=False):
        kept = 0
        for row in g.sort_values("score", ascending=False).itertuples(index=False):
            key = (str(series_id), str(event), int(row.step))
            if key in used:
                continue
            rows.append(
                {
                    "series_id": series_id,
                    "step": int(row.step),
                    "event": event,
                    "score": float(np.clip(float(row.score) * score_scale, 0, 1)),
                }
            )
            kept += 1
            if kept >= keep_per_event:
                break
    return pd.DataFrame(rows, columns=["series_id", "step", "event", "score"])


def pair_candidates(pred_df: pd.DataFrame, candidates: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cfg = (config or {}).get("postprocess", config or {})
    min_sleep = int(cfg.get("min_sleep_steps", 360))
    max_sleep = int(cfg.get("max_sleep_steps", 11520))
    max_pairs = int(cfg.get("max_pairs_per_series", cfg.get("max_events_per_series_event", 200)))
    sleep_weight = float(cfg.get("pair_sleep_weight", 0.25))
    invalid_penalty = float(cfg.get("pair_invalid_penalty", cfg.get("invalid_penalty", 0.0)))
    fallback_keep = int(cfg.get("pair_fallback_keep_per_event", 20))
    fallback_scale = float(cfg.get("pair_fallback_score_scale", 0.35))
    rows = []
    used: set[tuple[str, str, int]] = set()

    if candidates.empty:
        return _empty_events()

    for series_id, g_pred in pred_df.groupby("series_id", sort=False):
        g_pred = g_pred.sort_values("step").reset_index(drop=True)
        steps = g_pred["step"].to_numpy(dtype=int)
        if len(steps) == 0:
            continue
        sleep = g_pred.get("p_sleep", pd.Series(0, index=g_pred.index)).fillna(0).to_numpy(dtype=float)
        invalid = g_pred.get("p_invalid", pd.Series(0, index=g_pred.index)).fillna(0).to_numpy(dtype=float)
        c = candidates[candidates["series_id"] == series_id]
        onsets = c[c["event"] == "onset"].sort_values("score", ascending=False).head(max_pairs * 2)
        wakeups = c[c["event"] == "wakeup"].sort_values("score", ascending=False).head(max_pairs * 2)
        pair_rows = []
        for onset in onsets.itertuples(index=False):
            valid_wakeups = wakeups[
                (wakeups["step"].astype(int) > int(onset.step) + min_sleep)
                & (wakeups["step"].astype(int) < int(onset.step) + max_sleep)
            ]
            for wakeup in valid_wakeups.itertuples(index=False):
                left = int(np.searchsorted(steps, int(onset.step), side="left"))
                right = int(np.searchsorted(steps, int(wakeup.step), side="right"))
                if right <= left:
                    continue
                sleep_mean = float(sleep[left:right].mean())
                invalid_mean = float(invalid[left:right].mean())
                pair_score = (float(onset.score) + float(wakeup.score)) / 2.0
                pair_score = pair_score + sleep_weight * sleep_mean - invalid_penalty * invalid_mean
                pair_rows.append((float(pair_score), onset, wakeup))
        for score, onset, wakeup in sorted(pair_rows, key=lambda x: x[0], reverse=True)[:max_pairs]:
            onset_key = (str(series_id), "onset", int(onset.step))
            wakeup_key = (str(series_id), "wakeup", int(wakeup.step))
            if onset_key in used or wakeup_key in used:
                continue
            used.add(onset_key)
            used.add(wakeup_key)
            score = float(np.clip(score, 0, 1))
            rows.append({"series_id": series_id, "step": int(onset.step), "event": "onset", "score": score})
            rows.append({"series_id": series_id, "step": int(wakeup.step), "event": "wakeup", "score": score})

    paired = pd.DataFrame(rows, columns=["series_id", "step", "event", "score"])
    fallback = _fallback_rows(candidates, used, fallback_keep, fallback_scale)
    if paired.empty:
        return fallback
    if fallback.empty:
        return paired
    return pd.concat([paired, fallback], ignore_index=True)


def predictions_to_submission(pred_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    parts = [find_candidates(pred_df, event, config) for event in EVENTS]
    sub = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    cfg = (config or {}).get("postprocess", config or {})
    if bool(cfg.get("use_pair_filter", False)) and not sub.empty:
        sub = pair_candidates(pred_df, sub, config)
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
