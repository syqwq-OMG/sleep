import pandas as pd

from src.postprocess import predictions_to_submission, validate_submission


def test_submission_format_from_predictions():
    pred = pd.DataFrame(
        {
            "series_id": ["s"] * 20,
            "step": list(range(20)),
            "p_onset": [0.0] * 10 + [0.5] + [0.0] * 9,
            "p_wakeup": [0.0] * 15 + [0.4] + [0.0] * 4,
            "p_sleep": [0.2] * 20,
            "p_invalid": [0.0] * 20,
        }
    )
    sub = predictions_to_submission(pred, {"postprocess": {"peak_distance_steps": 1}})
    sub = validate_submission(sub)
    assert list(sub.columns) == ["row_id", "series_id", "step", "event", "score"]
    assert set(sub["event"]) == {"onset", "wakeup"}


def test_pair_postprocess_keeps_valid_pair():
    pred = pd.DataFrame(
        {
            "series_id": ["s"] * 200,
            "step": list(range(200)),
            "p_onset": [0.0] * 30 + [0.8] + [0.0] * 169,
            "p_wakeup": [0.0] * 120 + [0.7] + [0.0] * 79,
            "p_sleep": [0.1] * 31 + [0.9] * 90 + [0.1] * 79,
            "p_invalid": [0.0] * 200,
        }
    )
    cfg = {
        "postprocess": {
            "peak_distance_steps": 1,
            "score_threshold": 0.0001,
            "use_pair_filter": True,
            "min_sleep_steps": 10,
            "max_sleep_steps": 150,
            "pair_fallback_keep_per_event": 0,
        }
    }
    sub = predictions_to_submission(pred, cfg)
    assert set(sub["event"]) == {"onset", "wakeup"}
    assert sub.loc[sub["event"] == "onset", "step"].iloc[0] == 30
    assert sub.loc[sub["event"] == "wakeup", "step"].iloc[0] == 120
