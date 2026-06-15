import numpy as np
import pandas as pd

from src.labels import make_event_labels, make_sleep_labels


def test_event_label_peak_at_event():
    steps = np.arange(100)
    events = pd.DataFrame([{"series_id": "s", "event": "onset", "step": 50}])
    y = make_event_labels(steps, events, sigma_steps=10)["onset"]
    assert y[50] == y.max()
    assert y[50] > y[20]


def test_sleep_labels_only_complete_interval():
    steps = np.arange(100)
    events = pd.DataFrame(
        [
            {"series_id": "s", "event": "onset", "step": 20},
            {"series_id": "s", "event": "wakeup", "step": 60},
            {"series_id": "s", "event": "onset", "step": 80},
        ]
    )
    y, mask = make_sleep_labels(steps, events)
    assert y[20:61].sum() == 41
    assert mask[20:61].sum() == 41
    assert y[80:].sum() == 0
