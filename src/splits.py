"""Walk-forward (expanding-window) folds with a gap between train and test.

All quantities here are SLOT INDICES `t` (the forecast time of a row), on the
grid defined by src.labels.AlertGrid.

Why a gap, and how big. A training row at slot t has a label that looks
H slots ahead, i.e. it reads the alert state up to slot t+H. If the last
training row were allowed to sit right next to the first test row, its label
window would reach into the test period. So we require

        train_last_t + gap_slots < test_first_t        (strict), gap_slots >= H

which also guarantees  train_last_t + H < test_first_t : no training label
ever reads data from a slot inside the test period. (Test-row FEATURES may of
course read the training period: that is the past, which a real forecaster
has.) `Fold.__post_init__` refuses to construct a fold that violates this, so
a leaky split cannot exist; tests/test_splits.py checks it again on real
index arithmetic.

CUTOFF. The scraper writes an alert only after it has ended, so the newest
alerts of a snapshot are missing. `last_usable_t_idx` therefore drops the last
CUTOFF_MARGIN (12 h) of the file and also requires the whole label window
(t, t+H] to lie before that cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CUTOFF_MARGIN, SLOT


def last_usable_t_idx(
    last_record_end: pd.Timestamp,
    t0: pd.Timestamp,
    horizon_slots: int,
    margin: pd.Timedelta = CUTOFF_MARGIN,
) -> int:
    """Largest slot index t such that the label window (t, t+H] ends at or
    before `last_record_end - margin`."""
    usable_end = (last_record_end - margin).floor(SLOT)
    return int((usable_end - t0) // SLOT) - horizon_slots


@dataclass(frozen=True)
class Fold:
    name: str
    train_first_t: int   # inclusive (before any training-window cap)
    train_last_t: int    # inclusive
    test_first_t: int    # inclusive
    test_last_t: int     # inclusive
    gap_slots: int

    def __post_init__(self) -> None:
        if self.train_last_t + self.gap_slots >= self.test_first_t:
            raise ValueError(
                f"{self.name}: train_last_t ({self.train_last_t}) + gap ({self.gap_slots}) "
                f"must be < test_first_t ({self.test_first_t})"
            )
        if self.train_first_t > self.train_last_t:
            raise ValueError(f"{self.name}: empty training range")
        if self.test_first_t > self.test_last_t:
            raise ValueError(f"{self.name}: empty test range")

    def train_range(self, window_slots: int | None = None) -> tuple[int, int]:
        """(first, last) training slot, optionally capped to the most recent
        `window_slots` slots before train_last_t (the 'last 1-2 years' option)."""
        first = self.train_first_t
        if window_slots is not None:
            first = max(first, self.train_last_t - window_slots + 1)
        return first, self.train_last_t


def make_walk_forward_folds(
    last_t_idx: int,
    first_t_idx: int,
    n_folds: int,
    test_slots: int,
    gap_slots: int,
) -> list[Fold]:
    """`n_folds` contiguous, non-overlapping test windows of `test_slots` slots,
    the LAST one ending at `last_t_idx`; folds are returned oldest first and the
    final one is named "final" (all earlier ones are "tune1", "tune2", ...).
    Training is expanding: from `first_t_idx` up to `gap_slots + 1` slots
    before the fold's first test slot."""
    if n_folds < 2:
        raise ValueError("need at least one tuning fold and the final fold")
    folds: list[Fold] = []
    for k in range(n_folds):  # k = 0 is the final (latest) fold
        test_last = last_t_idx - k * test_slots
        test_first = test_last - test_slots + 1
        train_last = test_first - gap_slots - 1
        name = "final" if k == 0 else f"tune{n_folds - k}"
        folds.append(Fold(name, first_t_idx, train_last, test_first, test_last, gap_slots))
    return folds[::-1]


def mask_in_range(t_idx: np.ndarray, first: int, last: int) -> np.ndarray:
    return (t_idx >= first) & (t_idx <= last)
