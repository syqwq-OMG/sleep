from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import ensure_dir, iter_series, load_series, make_windows
from .features import build_features
from .models import SleepMambaUNet
from .postprocess import load_config, predictions_to_submission
from .train import _synthetic_data


@torch.no_grad()
def predict_series(model, feature_matrix, windows, device) -> np.ndarray:
    n, _ = feature_matrix.shape
    sums = np.zeros((n, 4), dtype="float64")
    counts = np.zeros(n, dtype="float64")
    model.eval()
    for w in windows:
        valid_len = w["valid_len"]
        x = np.zeros((len(w["mask"]), feature_matrix.shape[1]), dtype="float32")
        x[:valid_len] = feature_matrix[w["start_idx"] : w["end_idx"]]
        out = model(torch.from_numpy(x[None]).to(device))
        probs = torch.stack([torch.sigmoid(out[k])[0] for k in ["onset", "wakeup", "sleep", "invalid"]], dim=-1).cpu().numpy()
        sums[w["start_idx"] : w["end_idx"]] += probs[:valid_len]
        counts[w["start_idx"] : w["end_idx"]] += 1
    return (sums / np.maximum(counts[:, None], 1)).astype("float32")


def _load_test(config: dict, phase: str) -> pd.DataFrame:
    raw = Path(config["data"]["raw_dir"])
    name = config["data"]["test_series"] if phase == "test" else config["data"]["train_series"]
    path = raw / name
    if path.exists():
        df = load_series(path)
    else:
        df, _ = _synthetic_data(n_series=int(config.get("debug", {}).get("n_series", 2)), length=config["training"].get("window_size", 720))
    n_debug = config.get("debug", {}).get("n_series")
    if n_debug:
        keep = df["series_id"].drop_duplicates().astype(str).head(int(n_debug)).tolist()
        df = df[df["series_id"].astype(str).isin(keep)].reset_index(drop=True)
    return df


def run_infer(config: dict, checkpoints: list[str], phase: str, weights: list[float] | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    series = _load_test(config, phase)
    rows = []
    models = []
    feature_names = None
    if weights is None:
        weights = [1.0] * len(checkpoints)
    if len(weights) != len(checkpoints):
        raise ValueError("weights length must match checkpoints length")
    weights = np.asarray(weights, dtype="float64")
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive value")
    weights = weights / weight_sum
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device)
        feature_names = ckpt["feature_names"]
        mcfg = ckpt.get("config", config).get("model", config.get("model", {}))
        model = SleepMambaUNet(len(feature_names), mcfg.get("base_dim", 96), mcfg.get("num_heads", 4), mcfg.get("dropout", 0.1)).to(device)
        model.load_state_dict(ckpt["model"])
        models.append(model)
    if not models:
        raise ValueError("at least one checkpoint is required")
    for sid, g in iter_series(series):
        x, names = build_features(g, **config.get("features", {}))
        if feature_names and names != feature_names:
            raise ValueError("feature mismatch between checkpoint and inference data")
        windows = make_windows(g, config["training"].get("window_size", 17280), config["training"].get("stride", 4320), pad=True)
        pred = np.zeros((len(g), 4), dtype="float64")
        for model, weight in zip(models, weights):
            pred += weight * predict_series(model, x, windows, device)
        tmp = pd.DataFrame(
            {
                "series_id": sid,
                "step": g["step"].to_numpy(dtype=int),
                "p_onset": pred[:, 0].astype("float32"),
                "p_wakeup": pred[:, 1].astype("float32"),
                "p_sleep": pred[:, 2].astype("float32"),
                "p_invalid": pred[:, 3].astype("float32"),
            }
        )
        rows.append(tmp)
    out_root = ensure_dir(config.get("outputs", {}).get("root", "outputs"))
    pred_df = pd.concat(rows, ignore_index=True)
    pred_path = out_root / f"{phase}_pred.parquet"
    pred_df.to_parquet(pred_path, index=False)
    sub_dir = ensure_dir(out_root / "submissions")
    predictions_to_submission(pred_df, config).to_csv(sub_dir / "submission.csv", index=False)
    print(f"saved {pred_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--phase", default="test", choices=["test", "train"])
    args = parser.parse_args()
    run_infer(load_config(args.config), args.checkpoints, args.phase, args.weights)


if __name__ == "__main__":
    main()
