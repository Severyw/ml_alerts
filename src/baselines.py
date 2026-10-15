"""Baselines every model must be compared against (spec item 3).

All three are defined per MODE, because the two modes answer different
questions:

* persistence                 : "alert now -> alert later".
    continuation rows -> 1.0, onset rows -> 0.0. It is a CONSTANT within a
    mode, so its PR-AUC equals the positive rate by convention and its Brier
    score is simply the miss rate; it is reported for completeness, not as a
    ranking competitor.
* slot-of-day x day-of-week frequency (per oblast), fitted on the TRAINING
    rows of the fold only. Cells are shrunk toward the oblast's own training
    mean with a small pseudo-count so rare cells do not give 0 or 1.
* recent 24 h frequency       : uses only trailing features that are already
    in the feature row (so it cannot see anything after t).
      onset        : per-slot onset rate r = onsets in last 48 slots / 48,
                     p = 1 - (1 - r)^H  (chance of at least one onset in H slots
                     if onsets arrived at that constant rate).
      continuation : share of the last 48 slots in which the oblast was in
                     alert. This is a crude activity level, not a conditional
                     continuation rate; it is a deliberately weak baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import MODELLED_OBLASTS

N_SLOT, N_DOW = 48, 7


def persistence(mode: str, n: int) -> np.ndarray:
    return np.full(n, 1.0 if mode == "continuation" else 0.0)


def _cell_keys(oblast: np.ndarray, slot_of_day: np.ndarray, dow: np.ndarray):
    ob = pd.Categorical(oblast, categories=list(MODELLED_OBLASTS)).codes.astype(np.int64)
    if (ob < 0).any():
        raise ValueError("unknown oblast in baseline input")
    key = (ob * N_DOW + np.asarray(dow).astype(np.int64)) * N_SLOT + np.asarray(slot_of_day).astype(np.int64)
    return ob, key


class SlotDowFrequency:
    """Per-oblast mean of the target in each (slot-of-day, day-of-week) cell,
    shrunk toward the oblast mean: (sum_cell + a * oblast_mean) / (n_cell + a)."""

    def __init__(self, pseudo_count: float = 10.0) -> None:
        self.pseudo_count = pseudo_count
        self.table_: np.ndarray | None = None

    def fit(self, oblast, slot_of_day, dow, y) -> "SlotDowFrequency":
        y = np.asarray(y, dtype=float)
        ob, key = _cell_keys(np.asarray(oblast), slot_of_day, dow)
        n_obl = len(MODELLED_OBLASTS)
        n_cells = n_obl * N_DOW * N_SLOT
        n_cell = np.bincount(key, minlength=n_cells).astype(float)
        s_cell = np.bincount(key, weights=y, minlength=n_cells)
        n_o = np.bincount(ob, minlength=n_obl).astype(float)
        s_o = np.bincount(ob, weights=y, minlength=n_obl)
        overall = float(y.mean()) if len(y) else 0.0
        p_o = np.where(n_o > 0, s_o / np.maximum(n_o, 1.0), overall)
        prior = p_o[np.arange(n_cells) // (N_DOW * N_SLOT)]
        self.table_ = (s_cell + self.pseudo_count * prior) / (n_cell + self.pseudo_count)
        return self

    def predict(self, oblast, slot_of_day, dow) -> np.ndarray:
        if self.table_ is None:
            raise RuntimeError("call fit() first")
        _, key = _cell_keys(np.asarray(oblast), slot_of_day, dow)
        return self.table_[key]


def recent_frequency_24h(mode: str, onset_cnt_48, active_frac_48, horizon_slots: int) -> np.ndarray:
    if mode == "onset":
        r = np.clip(np.asarray(onset_cnt_48, dtype=float) / 48.0, 0.0, 1.0)
        return 1.0 - (1.0 - r) ** horizon_slots
    if mode == "continuation":
        return np.clip(np.asarray(active_frac_48, dtype=float), 0.0, 1.0)
    raise ValueError(mode)
