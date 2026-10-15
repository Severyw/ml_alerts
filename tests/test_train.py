"""Structural tests of the train pipeline on small synthetic data (fast)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MODELLED_OBLASTS, SLOT
from src.features import feature_columns
from src.splits import make_walk_forward_folds
from src.train import Dataset, ModelConfig, run_experiment

CFGS = [ModelConfig(10, 7, None), ModelConfig(20, 7, 100)]


def _synthetic(n_t=900, seed=0) -> Dataset:
    rng = np.random.default_rng(seed)
    obl = list(MODELLED_OBLASTS[:3])
    t = np.repeat(np.arange(n_t), len(obl))
    ob = np.tile(obl, n_t)
    X = pd.DataFrame(rng.random((len(t), len(feature_columns()))).astype("float32"), columns=feature_columns())
    X["slot_of_day"] = (t % 48).astype("float32")
    X["dow"] = ((t // 48) % 7).astype("float32")
    X["onset_cnt_48"] = rng.integers(0, 4, len(t)).astype("float32")
    X["active_frac_48"] = rng.random(len(t)).astype("float32")
    X["oblast"] = pd.Categorical(ob, categories=list(MODELLED_OBLASTS))
    mode = np.where(rng.random(len(t)) < 0.5, "onset", "continuation")
    y = (rng.random(len(t)) < 0.2 + 0.5 * X["active_frac_48"].to_numpy()).astype("int8")
    return Dataset(X, y, mode, t, ob, "1h", 2, pd.Timestamp("2026-01-01", tz="UTC"), 0, n_t - 1)


def _run(ds, log=lambda s: None):
    folds = make_walk_forward_folds(ds.last_t_idx, ds.first_t_idx, 3, 100, ds.horizon_slots)
    frozen = {}
    res, arr = run_experiment(ds, folds, CFGS, on_frozen=lambda m, c, t: frozen.update({m: c}),
                              compare_windows=False, log=log)
    return folds, res, arr, frozen


def test_final_fold_labels_cannot_influence_frozen_config_or_final_predictions():
    ds = _synthetic()
    folds, res, arr, frozen = _run(ds)
    final = folds[-1]
    ds2 = _synthetic()
    after = ds2.t_idx >= final.test_first_t - ds2.horizon_slots   # labels of the final test window (and its gap)
    ds2.y[after] = 1 - ds2.y[after]
    _, res2, arr2, frozen2 = _run(ds2)
    assert frozen == frozen2
    for m in ("onset", "continuation"):
        np.testing.assert_array_equal(arr[m]["preds"]["model"], arr2[m]["preds"]["model"])
        assert arr[m]["y"].tolist() != arr2[m]["y"].tolist()   # the flip really happened


def test_frozen_config_is_written_before_final_fold_is_scored():
    ds = _synthetic()
    order = []
    folds = make_walk_forward_folds(ds.last_t_idx, ds.first_t_idx, 3, 100, 2)
    run_experiment(ds, folds, CFGS, on_frozen=lambda m, c, t: order.append(("frozen", m)),
                   compare_windows=False, log=lambda s: order.append(("log", s)))
    assert [o for o in order if o[0] == "frozen"] == [("frozen", "onset"), ("frozen", "continuation")]


def test_every_fold_respects_gap_and_reports_pos_rate():
    ds = _synthetic()
    folds, res, _, _ = _run(ds)
    for m, r in res["modes"].items():
        for f, fo in zip(folds, r["folds"]):
            assert fo["used_for_tuning"] == (f.name != "final")
            for name, met in fo["metrics"].items():
                assert 0 <= met["pos_rate"] <= 1 and "pos_rate" in met
        assert r["folds"][-1]["fold"] == "final"
        assert not r["folds"][-1]["used_for_tuning"]
