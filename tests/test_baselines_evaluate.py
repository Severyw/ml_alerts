from __future__ import annotations

import numpy as np
import pytest

from src import baselines as bl
from src.config import MODELLED_OBLASTS
from src.evaluate import brier, calibration_table, expected_calibration_error, pr_auc, summarize

A, B = MODELLED_OBLASTS[0], MODELLED_OBLASTS[1]


def test_persistence_constants():
    assert (bl.persistence("continuation", 3) == 1.0).all()
    assert (bl.persistence("onset", 3) == 0.0).all()


def test_slot_dow_recovers_cell_rates_and_uses_only_fit_data():
    ob = np.array([A] * 4 + [B] * 4)
    slot = np.array([0, 0, 1, 1] * 2)
    dow = np.zeros(8, dtype=int)
    y = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    m = bl.SlotDowFrequency(pseudo_count=1e-9).fit(ob, slot, dow, y)
    np.testing.assert_allclose(m.predict(ob, slot, dow), y.astype(float), atol=1e-6)
    # unseen cell falls back to the oblast mean
    p = bl.SlotDowFrequency(pseudo_count=10).fit(ob, slot, dow, y).predict(np.array([A]), np.array([40]), np.array([3]))
    assert p[0] == pytest.approx(0.5)


def test_recent_frequency():
    p = bl.recent_frequency_24h("onset", np.array([0, 48]), np.array([0.0, 0.0]), 2)
    assert p[0] == 0.0 and p[1] == 1.0
    q = bl.recent_frequency_24h("continuation", np.array([0, 0]), np.array([0.25, 1.5]), 2)
    np.testing.assert_allclose(q, [0.25, 1.0])


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert brier(y, np.array([0, 0, 1, 1.0])) == 0.0
    assert pr_auc(y, np.array([.1, .2, .8, .9])) == 1.0
    assert pr_auc(y, np.full(4, .3)) == pytest.approx(0.5)      # constant score == positive rate
    assert np.isnan(pr_auc(np.zeros(4), np.arange(4.0)))
    s = summarize(y, np.full(4, .5))
    assert s["pos_rate"] == 0.5 and s["n"] == 4 and s["n_pos"] == 2 and s["ece"] == 0.0


def test_calibration_table_bins_and_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    t = calibration_table(y, p)
    assert t["n"].sum() == 20000 and len(t) == 10
    assert expected_calibration_error(t) < 0.03
    assert len(calibration_table(y, np.full(len(y), .2))) == 1
