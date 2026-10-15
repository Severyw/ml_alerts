"""Split-leakage tests. Central one: train max t + gap < validation min t."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.splits import Fold, last_usable_t_idx, make_walk_forward_folds, mask_in_range

H = 2


def test_fold_refuses_leaky_split():
    with pytest.raises(ValueError):
        Fold("bad", 0, 100, 101, 200, gap_slots=H)      # adjacent: label window reaches the test period
    with pytest.raises(ValueError):
        Fold("bad", 0, 100, 102, 200, gap_slots=H)      # train_last + gap == test_first -> still refused (strict)
    Fold("ok", 0, 100, 103, 200, gap_slots=H)           # 100 + 2 < 103


@pytest.mark.parametrize("gap", [2, 12])
@pytest.mark.parametrize("n_folds", [2, 3])
def test_split_leakage_train_max_plus_gap_below_validation_min(gap, n_folds):
    folds = make_walk_forward_folds(last_t_idx=20000, first_t_idx=336, n_folds=n_folds, test_slots=1344, gap_slots=gap)
    assert len(folds) == n_folds and folds[-1].name == "final" and folds[-1].test_last_t == 20000
    for f in folds:
        for window in (None, 365 * 48, 730 * 48):
            first, last = f.train_range(window)
            assert last + gap < f.test_first_t          # the requirement
            assert last + gap >= f.test_first_t - 1     # ...and the gap is not wasteful
            assert first >= f.train_first_t
    # test windows are disjoint, ordered, and later folds never test on earlier training end
    for a, b in zip(folds, folds[1:]):
        assert a.test_last_t < b.test_first_t


def test_label_windows_of_training_rows_stay_before_test_rows():
    """Row-level version: the label of train row t reads slots t+1..t+H."""
    f = make_walk_forward_folds(5000, 100, 3, 500, H)[-1]
    t = np.arange(0, 6000)
    tr = mask_in_range(t, *f.train_range())
    te = mask_in_range(t, f.test_first_t, f.test_last_t)
    assert t[tr].max() + H < t[te].min()
    assert not (tr & te).any()


def test_cutoff_keeps_label_window_before_12h_margin():
    t0 = pd.Timestamp("2022-03-15", tz="UTC")
    last_end = pd.Timestamp("2026-09-07 05:38:32", tz="UTC")
    for h in (2, 12):
        t = last_usable_t_idx(last_end, t0, h)
        window_end = t0 + (t + h) * pd.Timedelta("30min")
        assert window_end <= last_end - pd.Timedelta("12h")
        assert t0 + (t + 1 + h) * pd.Timedelta("30min") > last_end - pd.Timedelta("12h")


def test_need_a_tuning_fold():
    with pytest.raises(ValueError):
        make_walk_forward_folds(1000, 0, 1, 100, H)
