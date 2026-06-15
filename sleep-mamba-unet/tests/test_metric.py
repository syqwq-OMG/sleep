import pandas as pd

from src.metric import match_event_predictions, score_events


def test_match_event_predictions_one_tp_one_fp():
    sol = pd.DataFrame([{"series_id": "s", "event": "onset", "step": 100}])
    pred = pd.DataFrame(
        [
            {"series_id": "s", "event": "onset", "step": 101, "score": 0.9},
            {"series_id": "s", "event": "onset", "step": 300, "score": 0.8},
        ]
    )
    assert match_event_predictions(sol, pred, 12) == (1, 1, 0)


def test_score_events_perfect_is_one():
    sol = pd.DataFrame(
        [
            {"series_id": "s", "event": "onset", "step": 100},
            {"series_id": "s", "event": "wakeup", "step": 200},
        ]
    )
    pred = pd.DataFrame(
        [
            {"series_id": "s", "event": "onset", "step": 100, "score": 0.9},
            {"series_id": "s", "event": "wakeup", "step": 200, "score": 0.8},
        ]
    )
    assert score_events(sol, pred) == 1.0
