from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


def load_series(
    path: str | Path,
    columns: list[str] | None = None,
    series_ids: list[str] | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    filters = None
    if series_ids:
        filters = [("series_id", "in", list(map(str, series_ids)))]
    df = pd.read_parquet(path, columns=columns, filters=filters)
    if "series_id" not in df.columns or "step" not in df.columns:
        raise ValueError("series parquet must contain series_id and step")
    return df.sort_values(["series_id", "step"]).reset_index(drop=True)


def load_events(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"series_id", "event", "step"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"events csv missing columns: {sorted(missing)}")
    return df.dropna(subset=["series_id", "event", "step"]).copy()


def iter_series(df: pd.DataFrame) -> Iterator[tuple[str, pd.DataFrame]]:
    for series_id, g in df.groupby("series_id", sort=False):
        yield str(series_id), g.sort_values("step").reset_index(drop=True)


def make_windows(
    series_df: pd.DataFrame,
    window_size: int = 17280,
    stride: int = 4320,
    pad: bool = True,
) -> list[dict]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    n = len(series_df)
    series_id = str(series_df["series_id"].iloc[0]) if n and "series_id" in series_df else ""
    steps = series_df["step"].to_numpy() if "step" in series_df else np.arange(n)
    if n == 0:
        return []

    starts = list(range(0, max(n - window_size + 1, 1), stride))
    if starts[-1] != max(n - window_size, 0):
        starts.append(max(n - window_size, 0))

    windows = []
    for start_idx in starts:
        end_idx = min(start_idx + window_size, n)
        valid_len = end_idx - start_idx
        if valid_len < window_size and not pad:
            continue
        pad_len = window_size - valid_len
        mask = np.zeros(window_size, dtype=bool)
        mask[:valid_len] = True
        windows.append(
            {
                "series_id": series_id,
                "start_step": int(steps[start_idx]),
                "end_step": int(steps[end_idx - 1]),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "valid_len": int(valid_len),
                "pad_len": int(pad_len),
                "mask": mask,
            }
        )
    return windows


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
