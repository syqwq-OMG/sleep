from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import ensure_dir, iter_series, load_events, load_series, make_windows
from .features import build_features
from .folds import make_group_folds
from .labels import make_training_targets
from .losses import compute_loss
from .models import SleepMambaUNet
from .postprocess import load_config, predictions_to_submission


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _synthetic_data(n_series: int = 4, length: int = 1440) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, events = [], []
    for i in range(n_series):
        sid = f"debug_{i}"
        step = np.arange(length)
        anglez = 20 * np.sin(step / 150) + np.random.normal(0, 3, length)
        enmo = np.abs(np.random.normal(0.02, 0.01, length))
        onset, wakeup = 360, 1080
        enmo[onset:wakeup] *= 0.4
        rows.append(pd.DataFrame({"series_id": sid, "step": step, "anglez": anglez, "enmo": enmo}))
        events += [
            {"series_id": sid, "night": 0, "event": "onset", "step": onset},
            {"series_id": sid, "night": 0, "event": "wakeup", "step": wakeup},
        ]
    return pd.concat(rows, ignore_index=True), pd.DataFrame(events)


def _limit_windows(windows: list[dict], targets: dict[str, np.ndarray], config: dict, split: str) -> list[dict]:
    max_windows = config.get("debug", {}).get("max_windows_per_series")
    if not max_windows or len(windows) <= int(max_windows):
        return windows
    max_windows = int(max_windows)
    if split != "train" or config.get("debug", {}).get("window_sampling", "event_focused") != "event_focused":
        idx = np.linspace(0, len(windows) - 1, max_windows).round().astype(int)
        return [windows[i] for i in idx]

    event_scores = []
    for i, w in enumerate(windows):
        start, end = w["start_idx"], w["end_idx"]
        score = float(targets["y_onset"][start:end].max() + targets["y_wakeup"][start:end].max())
        event_scores.append((score, i))

    n_event = max(1, int(round(max_windows * float(config.get("debug", {}).get("event_window_fraction", 0.75)))))
    picked = [i for score, i in sorted(event_scores, reverse=True) if score > 0][:n_event]
    if len(picked) < n_event:
        picked.extend([i for _, i in sorted(event_scores, reverse=True) if i not in picked][: n_event - len(picked)])

    spread = np.linspace(0, len(windows) - 1, max_windows).round().astype(int).tolist()
    for i in spread:
        if len(picked) >= max_windows:
            break
        if i not in picked:
            picked.append(i)
    return [windows[i] for i in sorted(set(picked))[:max_windows]]


class WindowDataset(Dataset):
    def __init__(self, series_df: pd.DataFrame, events_df: pd.DataFrame, config: dict, series_ids: set[str], split: str = "train"):
        self.items = []
        self.feature_names = None
        self.lazy_windows = bool(config.get("training", {}).get("lazy_windows", False))
        self.series_cache = {}
        self.events_df = events_df
        self.config = config
        tcfg = config.get("training", {})
        for sid, g in iter_series(series_df[series_df["series_id"].astype(str).isin(series_ids)]):
            targets = make_training_targets(g, events_df, config)
            windows = make_windows(g, tcfg.get("window_size", 17280), tcfg.get("stride", 4320), pad=True)
            windows = _limit_windows(windows, targets, config, split)
            if self.lazy_windows:
                if self.feature_names is None and windows:
                    sample = g.iloc[windows[0]["start_idx"] : windows[0]["end_idx"]].reset_index(drop=True)
                    _, self.feature_names = build_features(sample, **config.get("features", {}))
                self.series_cache[sid] = g.reset_index(drop=True)
                for w in windows:
                    self.items.append((sid, w))
            else:
                x, names = build_features(g, **config.get("features", {}))
                self.feature_names = names
                for w in windows:
                    self.items.append((sid, g["step"].to_numpy(), x, targets, w))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.lazy_windows:
            sid, w = self.items[idx]
            g = self.series_cache[sid]
            start, end, valid_len = w["start_idx"], w["end_idx"], w["valid_len"]
            win_g = g.iloc[start:end].reset_index(drop=True)
            x, names = build_features(win_g, **self.config.get("features", {}))
            if self.feature_names is None:
                self.feature_names = names
            targets = make_training_targets(win_g, self.events_df, self.config)
            win_x = np.zeros((len(w["mask"]), x.shape[1]), dtype="float32")
            win_x[:valid_len] = x[:valid_len]
            out = {
                "x": torch.from_numpy(win_x),
                "mask_valid": torch.from_numpy(w["mask"].astype("float32")),
                "series_id": sid,
                "steps": torch.from_numpy(
                    np.pad(win_g["step"].to_numpy(), (0, len(w["mask"]) - valid_len), constant_values=-1).astype("int64")
                ),
            }
            for key, arr in targets.items():
                y = np.zeros(len(w["mask"]), dtype="float32")
                y[:valid_len] = arr[:valid_len]
                out[key] = torch.from_numpy(y)
            return out

        sid, steps, x, targets, w = self.items[idx]
        start, end, valid_len = w["start_idx"], w["end_idx"], w["valid_len"]
        win_x = np.zeros((len(w["mask"]), x.shape[1]), dtype="float32")
        win_x[:valid_len] = x[start:end]
        out = {
            "x": torch.from_numpy(win_x),
            "mask_valid": torch.from_numpy(w["mask"].astype("float32")),
            "series_id": sid,
            "steps": torch.from_numpy(np.pad(steps[start:end], (0, len(w["mask"]) - valid_len), constant_values=-1).astype("int64")),
        }
        for key, arr in targets.items():
            y = np.zeros(len(w["mask"]), dtype="float32")
            y[:valid_len] = arr[start:end]
            out[key] = torch.from_numpy(y)
        return out


def _load_or_synthetic(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = Path(config["data"]["raw_dir"])
    series_path = raw / config["data"]["train_series"]
    events_path = raw / config["data"]["train_events"]
    n_debug = config.get("debug", {}).get("n_series")
    if series_path.exists() and events_path.exists():
        events = load_events(events_path)
        debug_ids = None
        if n_debug:
            candidates = events["series_id"].dropna().astype(str).drop_duplicates().tolist()
            if config.get("debug", {}).get("series_sample", "random") == "first":
                debug_ids = candidates[: int(n_debug)]
            else:
                rng = np.random.default_rng(int(config.get("seed", 42)))
                debug_ids = rng.permutation(candidates).tolist()[: int(n_debug)]
        series = load_series(series_path, series_ids=debug_ids)
        if debug_ids:
            events = events[events["series_id"].astype(str).isin(debug_ids)].reset_index(drop=True)
    else:
        n = int(config.get("debug", {}).get("n_series", 4))
        series, events = _synthetic_data(n_series=n)
    if n_debug and not series_path.exists():
        keep = series["series_id"].drop_duplicates().astype(str).head(int(n_debug)).tolist()
        series = series[series["series_id"].astype(str).isin(keep)].reset_index(drop=True)
        events = events[events["series_id"].astype(str).isin(keep)].reset_index(drop=True)
    return series, events


def train_one_fold(config: dict, fold: int):
    set_seed(int(config.get("seed", 42)))
    series, events = _load_or_synthetic(config)
    folds = make_group_folds(series["series_id"].astype(str).unique(), config["training"].get("folds", 5), config.get("seed", 42))
    train_ids = {sid for sid, f in folds.items() if f != fold}
    valid_ids = {sid for sid, f in folds.items() if f == fold}
    if not train_ids:
        train_ids = valid_ids
    train_ds = WindowDataset(series, events, config, train_ids, split="train")
    valid_ds = WindowDataset(series, events, config, valid_ids, split="valid")
    if len(train_ds) == 0:
        raise RuntimeError("no training windows")
    feature_names = train_ds.feature_names or valid_ds.feature_names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = config.get("model", {})
    model = SleepMambaUNet(
        len(feature_names),
        model_cfg.get("base_dim", 96),
        model_cfg.get("num_heads", 4),
        model_cfg.get("dropout", 0.1),
        model_cfg.get("boundary_mix", 0.0),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["training"].get("lr", 5e-4), weight_decay=config["training"].get("weight_decay", 0.01))
    loader = DataLoader(train_ds, batch_size=config["training"].get("batch_size", 2), shuffle=True, num_workers=0)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["training"].get("amp", True)) and device.type == "cuda")
    for epoch in range(int(config["training"].get("epochs", 1))):
        model.train()
        pbar = tqdm(loader, desc=f"fold {fold} epoch {epoch}")
        for batch in pbar:
            x = batch["x"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k.startswith("y_") or k.startswith("mask_")}
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss = compute_loss(model(x), targets, config)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"].get("grad_clip", 1.0))
            scaler.step(opt)
            scaler.update()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    out_root = ensure_dir(config.get("outputs", {}).get("root", "outputs"))
    model_dir = ensure_dir(out_root / "models")
    ckpt = model_dir / f"fold{fold}.pt"
    torch.save({"model": model.state_dict(), "feature_names": feature_names, "config": config}, ckpt)
    (model_dir / f"fold{fold}_meta.json").write_text(json.dumps({"feature_names": feature_names}, indent=2))
    pred = predict_dataset(model, valid_ds, device)
    pred_path = out_root / "oof_pred.parquet"
    pred.to_parquet(pred_path, index=False)
    sub_dir = ensure_dir(out_root / "submissions")
    predictions_to_submission(pred, config).to_csv(sub_dir / "debug_submission.csv", index=False)
    print(f"saved {ckpt}")
    print(f"saved {pred_path}")


@torch.no_grad()
def predict_dataset(model, dataset, device) -> pd.DataFrame:
    model.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        probs = {k: torch.sigmoid(v).cpu().numpy()[0] for k, v in model(batch["x"].to(device)).items()}
        steps = batch["steps"].numpy()[0]
        sid = batch["series_id"][0]
        m = steps >= 0
        for i in np.where(m)[0]:
            rows.append(
                {
                    "series_id": sid,
                    "step": int(steps[i]),
                    "p_onset": float(probs["onset"][i]),
                    "p_wakeup": float(probs["wakeup"][i]),
                    "p_sleep": float(probs["sleep"][i]),
                    "p_invalid": float(probs["invalid"][i]),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["series_id", "step", "p_onset", "p_wakeup", "p_sleep", "p_invalid"])
    return pd.DataFrame(rows).groupby(["series_id", "step"], as_index=False).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    fold = int(args.fold if args.fold is not None else cfg["training"].get("fold", 0))
    train_one_fold(cfg, fold)


if __name__ == "__main__":
    main()
