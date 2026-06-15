from __future__ import annotations

import numpy as np
import pandas as pd


EVENTS = ("onset", "wakeup")


def make_event_labels(
    steps: np.ndarray,
    events_df: pd.DataFrame,
    sigma_steps: int | list[int],
    sigma_weights: list[float] | None = None,
) -> dict[str, np.ndarray]:
    steps = np.asarray(steps, dtype=float)
    sigmas = np.atleast_1d(np.asarray(sigma_steps, dtype=float))
    if sigma_weights is None:
        weights = np.ones(len(sigmas), dtype=float) / len(sigmas)
    else:
        weights = np.asarray(sigma_weights, dtype=float)
        weights = weights / max(weights.sum(), 1e-12)
    out = {event: np.zeros(len(steps), dtype="float32") for event in EVENTS}
    if events_df is None or events_df.empty:
        return out
    clean = events_df.dropna(subset=["event", "step"])
    for event in EVENTS:
        y = np.zeros(len(steps), dtype=float)
        for event_step in clean.loc[clean["event"] == event, "step"].astype(float):
            dist = steps - event_step
            for sigma, weight in zip(sigmas, weights):
                y += weight * np.exp(-0.5 * (dist / sigma) ** 2)
        out[event] = np.clip(y, 0, 1).astype("float32")
    return out


def make_sleep_labels(steps: np.ndarray, events_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    steps = np.asarray(steps)
    y_sleep = np.zeros(len(steps), dtype="float32")
    mask_sleep = np.zeros(len(steps), dtype="float32")
    if events_df is None or events_df.empty:
        return y_sleep, mask_sleep
    ev = events_df.dropna(subset=["event", "step"]).sort_values("step")
    pending_onset = None
    for row in ev.itertuples(index=False):
        event = getattr(row, "event")
        step = getattr(row, "step")
        if event == "onset":
            pending_onset = float(step)
        elif event == "wakeup" and pending_onset is not None and float(step) > pending_onset:
            m = (steps >= pending_onset) & (steps <= float(step))
            y_sleep[m] = 1.0
            mask_sleep[m] = 1.0
            pending_onset = None
    return y_sleep, mask_sleep


def make_training_targets(series_df: pd.DataFrame, events_df: pd.DataFrame, config: dict) -> dict[str, np.ndarray]:
    steps = series_df["step"].to_numpy()
    labels_cfg = config.get("labels", config)
    sigma_steps = labels_cfg.get("sigma_steps", [36, 120, 240])
    sigma_weights = labels_cfg.get("sigma_weights", [0.5, 0.3, 0.2])
    series_id = series_df["series_id"].iloc[0]
    ev = events_df[events_df["series_id"] == series_id] if events_df is not None and not events_df.empty else pd.DataFrame()
    event_labels = make_event_labels(steps, ev, sigma_steps, sigma_weights)
    y_sleep, mask_sleep = make_sleep_labels(steps, ev)
    mask_valid = np.ones(len(steps), dtype="float32")
    mask_event = np.ones(len(steps), dtype="float32") if not ev.empty else np.zeros(len(steps), dtype="float32")
    return {
        "y_onset": event_labels["onset"],
        "y_wakeup": event_labels["wakeup"],
        "y_sleep": y_sleep,
        "mask_event": mask_event,
        "mask_sleep": mask_sleep,
        "mask_valid": mask_valid,
    }
