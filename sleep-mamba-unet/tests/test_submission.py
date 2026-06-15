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
