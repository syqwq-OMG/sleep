from __future__ import annotations

import numpy as np
import pandas as pd


TOLERANCES_STEPS = [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
EVENTS = ("onset", "wakeup")


def _clean_solution(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["series_id", "event", "step"]).copy()


def match_event_predictions(
    solution_event_df: pd.DataFrame,
    pred_event_df: pd.DataFrame,
    tolerance_steps: int,
) -> tuple[int, int, int]:
    sol = _clean_solution(solution_event_df)
    pred = pred_event_df.dropna(subset=["series_id", "event", "step", "score"]).copy()
    pred = pred.sort_values("score", ascending=False)
    matched: set[int] = set()
    tp = 0
    for p in pred.itertuples():
        candidates = sol[
            (sol["series_id"] == p.series_id)
            & (sol["event"] == p.event)
            & ((sol["step"].astype(float) - float(p.step)).abs() <= tolerance_steps)
        ]
        best_idx = None
        best_dist = None
        for idx, row in candidates.iterrows():
            if idx in matched:
                continue
            dist = abs(float(row["step"]) - float(p.step))
            if best_dist is None or dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is not None:
            matched.add(best_idx)
            tp += 1
    fp = len(pred) - tp
    fn = len(sol) - tp
    return tp, fp, fn


def _average_precision(solution: pd.DataFrame, pred: pd.DataFrame, tolerance_steps: int) -> float:
    n_gt = len(_clean_solution(solution))
    if n_gt == 0:
        return 0.0
    pred = pred.dropna(subset=["series_id", "event", "step", "score"]).sort_values("score", ascending=False)
    matched: set[int] = set()
    tp_flags = []
    for p in pred.itertuples():
        candidates = solution[
            (solution["series_id"] == p.series_id)
            & (solution["event"] == p.event)
            & ((solution["step"].astype(float) - float(p.step)).abs() <= tolerance_steps)
        ]
        best_idx = None
        best_dist = None
        for idx, row in candidates.iterrows():
            if idx in matched:
                continue
            dist = abs(float(row["step"]) - float(p.step))
            if best_dist is None or dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is None:
            tp_flags.append(0.0)
        else:
            matched.add(best_idx)
            tp_flags.append(1.0)
    if not tp_flags:
        return 0.0
    tp = np.cumsum(tp_flags)
    fp = np.cumsum(1 - np.asarray(tp_flags))
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_gt
    recall = np.r_[0.0, recall, 1.0]
    precision = np.r_[1.0, precision, 0.0]
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def score_events(solution_df: pd.DataFrame, submission_df: pd.DataFrame, tolerances_steps=None) -> float:
    tolerances_steps = tolerances_steps or TOLERANCES_STEPS
    scores = []
    solution = _clean_solution(solution_df)
    pred = submission_df.copy()
    for event in EVENTS:
        sol_e = solution[solution["event"] == event]
        pred_e = pred[pred["event"] == event]
        for tol in tolerances_steps:
            scores.append(_average_precision(sol_e, pred_e, int(tol)))
    return float(np.mean(scores)) if scores else 0.0
