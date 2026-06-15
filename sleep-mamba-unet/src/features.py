from __future__ import annotations

import numpy as np
import pandas as pd


BASE_FEATURES = ["anglez", "enmo"]


def _rolling(s: pd.Series, window: int, fn: str) -> pd.Series:
    r = s.rolling(window=window, center=True, min_periods=max(1, window // 4))
    if fn == "mean":
        return r.mean()
    if fn == "std":
        return r.std()
    if fn == "median":
        return r.median()
    raise ValueError(fn)


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        hour = ts.dt.hour.fillna(0).to_numpy() + ts.dt.minute.fillna(0).to_numpy() / 60.0
        minute = ts.dt.minute.fillna(0).to_numpy()
        weekday = ts.dt.dayofweek.fillna(0).to_numpy()
    else:
        step = df["step"].to_numpy(dtype=float)
        hour = (step * 5 / 3600.0) % 24
        minute = (step * 5 / 60.0) % 60
        weekday = np.zeros_like(hour)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    out["minute_cos"] = np.cos(2 * np.pi * minute / 60)
    out["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    return out


def _quality_flags(g: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=g.index)
    std_anglez = _rolling(g["anglez"], 360, "std").fillna(0)
    std_enmo = _rolling(g["enmo"], 360, "std").fillna(0)
    out["non_wear_flag"] = ((std_anglez < 0.02) & (std_enmo < 0.001)).astype("float32")
    out["repeating_flag"] = 0.0
    return out


def build_features(
    df: pd.DataFrame,
    rolling_windows_steps: list[int] | None = None,
    use_time_features: bool = True,
    use_quality_flags: bool = True,
    normalize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    rolling_windows_steps = rolling_windows_steps or [12, 60, 360, 1440]
    frames = []
    for _, g in df.groupby("series_id", sort=False):
        g = g.sort_values("step")
        f = pd.DataFrame(index=g.index)
        anglez = g["anglez"].astype(float).interpolate(limit_direction="both").fillna(0)
        enmo = g["enmo"].astype(float).clip(lower=0).interpolate(limit_direction="both").fillna(0)
        f["anglez"] = anglez
        f["enmo"] = enmo
        f["log1p_enmo"] = np.log1p(enmo)
        f["d_anglez"] = anglez.diff().fillna(0)
        f["abs_d_anglez"] = f["d_anglez"].abs()
        f["d_enmo"] = enmo.diff().fillna(0)
        f["abs_d_enmo"] = f["d_enmo"].abs()
        for w in rolling_windows_steps:
            f[f"anglez_mean_{w}"] = _rolling(anglez, w, "mean")
            f[f"anglez_std_{w}"] = _rolling(anglez, w, "std")
            f[f"enmo_mean_{w}"] = _rolling(enmo, w, "mean")
            f[f"enmo_std_{w}"] = _rolling(enmo, w, "std")
        f["abs_d_anglez_med_60"] = _rolling(f["abs_d_anglez"], 60, "median")
        if use_time_features:
            f = pd.concat([f, _time_features(g)], axis=1)
        if use_quality_flags:
            f = pd.concat([f, _quality_flags(g)], axis=1)
        f = f.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        if normalize:
            for col in f.columns:
                if col.endswith("_flag") or col.endswith("_sin") or col.endswith("_cos"):
                    continue
                values = f[col].to_numpy(dtype=float)
                med = np.nanmedian(values)
                q75, q25 = np.nanpercentile(values, [75, 25])
                iqr = max(float(q75 - q25), 1e-6)
                f[col] = np.clip((values - med) / iqr, -10, 10)
        frames.append(f)
    feat_df = pd.concat(frames).loc[df.index]
    return feat_df.astype("float32").to_numpy(), list(feat_df.columns)
