from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold


def make_group_folds(series_ids, n_splits: int = 5, seed: int = 42) -> dict[str, int]:
    unique = np.array(sorted(set(map(str, series_ids))))
    if len(unique) < 2:
        return {str(s): 0 for s in unique}
    n_splits = max(2, min(n_splits, len(unique)))
    dummy = np.zeros(len(unique))
    groups = unique
    folds = {}
    for fold, (_, va_idx) in enumerate(GroupKFold(n_splits=n_splits).split(dummy, dummy, groups)):
        for sid in unique[va_idx]:
            folds[str(sid)] = fold
    return folds
