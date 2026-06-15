from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_single_night(series_df: pd.DataFrame, pred_df: pd.DataFrame | None = None, events_df: pd.DataFrame | None = None, out: str | Path | None = None):
    sid = series_df["series_id"].iloc[0]
    fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(series_df["step"], series_df["anglez"], lw=0.8)
    axes[0].set_ylabel("anglez")
    axes[1].plot(series_df["step"], series_df["enmo"], lw=0.8)
    axes[1].set_ylabel("enmo")
    if pred_df is not None and not pred_df.empty:
        p = pred_df[pred_df["series_id"] == sid]
        axes[2].plot(p["step"], p["p_sleep"], label="sleep")
        axes[2].set_ylabel("p_sleep")
        axes[3].plot(p["step"], p["p_onset"], label="onset")
        axes[3].plot(p["step"], p["p_wakeup"], label="wakeup")
        axes[3].legend()
    if events_df is not None and not events_df.empty:
        for ax in axes:
            for e in events_df[events_df["series_id"] == sid].itertuples():
                ax.axvline(e.step, color="tab:red" if e.event == "onset" else "tab:green", alpha=0.6)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    return fig


def plot_event_time_distribution(events_df: pd.DataFrame, out: str | Path | None = None):
    ev = events_df.dropna(subset=["step"]).copy()
    ev["hour"] = (ev["step"] * 5 / 3600) % 24
    fig, ax = plt.subplots(figsize=(8, 4))
    for event, g in ev.groupby("event"):
        ax.hist(g["hour"], bins=24, alpha=0.5, label=event)
    ax.set_xlabel("hour")
    ax.legend()
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig
