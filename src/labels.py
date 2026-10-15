"""Builds the per-oblast alert grid and the onset/continuation labels on it.

This module only computes the TARGET side (what we are trying to predict).
Feature engineering (src/features.py, not written yet) consumes the same
`AlertGrid` to build inputs, and must only ever read `grid.state[:, :=t_idx]`
and `grid.onset_cum[:, :=t_idx]` -- i.e. up to and including the current
slot, never beyond it. `build_label_frame` below is the one place that is
allowed to look past t (by construction: that's what a label is), and it
does so by exactly `horizon_slots`, never more.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import SLOT
from src.data_io import union_intervals


@dataclass(frozen=True)
class AlertGrid:
    """A per-oblast grid of 30-minute slot boundaries t0, t0+SLOT, ...,
    t0+n_steps*SLOT (n_steps+1 boundaries total, indices 0..n_steps).

    state[o, i]      : bool, True iff oblast `oblasts[o]` has an alert active
                        at slot boundary i (see build_grid for the ceil
                        convention that makes this leakage-safe).
    onset_cum[o, i]   : int, cumulative count of NEW onsets in oblast o at
                        slot boundaries 0..i inclusive. onset_cum[o, j] -
                        onset_cum[o, i] (for j > i) is the number of onsets
                        in slots (i, j].
    """

    oblasts: list[str]
    t0: pd.Timestamp
    n_steps: int
    state: np.ndarray
    onset_cum: np.ndarray

    def index_of(self, t: pd.Timestamp) -> int:
        """Slot index of timestamp `t`. `t` must already sit on a slot
        boundary (i.e. t == t.ceil(SLOT)); this is NOT a rounding helper."""
        offset = (t - self.t0) / SLOT
        idx = round(offset)
        if abs(offset - idx) > 1e-9:
            raise ValueError(f"{t} is not on a {SLOT} slot boundary relative to t0={self.t0}")
        return idx


def build_grid(
    df: pd.DataFrame,
    t0: pd.Timestamp,
    n_steps: int,
    key: str = "oblast",
    oblasts: Sequence[str] | None = None,
) -> AlertGrid:
    """Build a state/onset grid from raw (possibly overlapping, per-raion)
    alert rows, covering slot boundaries t0 .. t0 + n_steps*SLOT.

    `oblasts`: the fixed row universe of the grid. ALWAYS pass it in real
    use (src.config.MODELLED_OBLASTS). If omitted, the universe is whatever
    oblasts happen to have events in `df`, which makes the array shape
    depend on the data -- deleting or perturbing events (as the leakage
    tests do) could then silently drop a row. An oblast with no events gets
    an all-False row. Events for oblasts outside the universe raise.

    CEIL CONVENTION (agreed spec). A raw event at time x is visible to the
    grid only from slot ceil(x) onward:
        x = 16:20  -> ceil -> 16:30   (not visible yet at t=16:00)
        x = 16:30  -> ceil -> 16:30   (visible exactly at t=16:30, not before)
        x = 16:31  -> ceil -> 17:00
    This is what keeps the grid leakage-safe for forecasting at any slot
    boundary t: nothing with a raw timestamp strictly after t can affect
    state[:, i] for any i with t0 + i*SLOT <= t, because ceil(x) > t
    whenever x > t (ceil never rounds down). See
    tests/test_labels.py::test_leakage_state_at_t_is_invariant_to_future_events.

    ZERO-SLOT ALERTS. If an episode's start and end ceil to the SAME slot
    (e.g. 16:05-16:25, both -> 16:30), it contributes an onset at that slot
    but `state` is never True for it -- per spec, it must still count as an
    onset. This falls out automatically: `onset` increments unconditionally
    at i0, while `state`'s delta is only touched when i1 > i0. See
    tests/test_labels.py::test_zero_slot_alert_counts_as_onset_but_never_active.
    """
    episodes = union_intervals(df, key=key)
    if oblasts is None:
        oblasts = sorted(episodes[key].unique())
    else:
        oblasts = list(oblasts)
        unknown = set(episodes[key].unique()) - set(oblasts)
        if unknown:
            raise ValueError(f"df has events for oblasts outside the declared universe: {sorted(unknown)}")
    idx = {o: i for i, o in enumerate(oblasts)}

    # +2 padding: one slot of headroom for an episode ending exactly at
    # n_steps, one more so `onset_cum[:, n_steps+1]` is always a valid index
    # (build_label_frame never reads it, but build_grid must not crash
    # constructing it for episodes that end right at the edge).
    delta = np.zeros((len(oblasts), n_steps + 2), dtype=np.int32)
    onset = np.zeros((len(oblasts), n_steps + 2), dtype=np.int32)

    for o, s, e in zip(episodes[key], episodes.started_at, episodes.finished_at):
        i0 = int(np.ceil((s - t0) / SLOT))
        i1 = int(np.ceil((e - t0) / SLOT))
        if i0 < 0 or i0 > n_steps:
            continue  # episode starts outside the requested grid window
        i1 = min(i1, n_steps + 1)
        onset[idx[o], i0] += 1
        if i1 > i0:
            delta[idx[o], i0] += 1
            if i1 <= n_steps:
                delta[idx[o], i1] -= 1

    state = np.cumsum(delta, axis=1) > 0
    onset_cum = np.cumsum(onset, axis=1)
    return AlertGrid(oblasts=oblasts, t0=t0, n_steps=n_steps, state=state, onset_cum=onset_cum)


def build_label_frame(
    grid: AlertGrid,
    t_start_idx: int,
    t_end_idx: int,
    horizons: dict[str, int],
) -> pd.DataFrame:
    """One row per (oblast, t) for slot index t in [t_start_idx, t_end_idx).

    mode is fixed at t, independent of horizon (per spec: "look at the state
    at t. If no alert -> return P(onset). If alert active -> return
    P(continues)."):
        mode = "onset"        if no alert active at t
        mode = "continuation" if an alert is active at t

    target_<h> is the label matching that row's mode, for horizon h:
        mode == "onset"        -> target_<h> = new alert starts in (t, t+h]
        mode == "continuation" -> target_<h> = alert still active at exactly t+h

    GAP DISCIPLINE. This raises rather than silently truncating or wrapping
    if [t_start_idx, t_end_idx) together with the largest horizon would read
    past the grid -- i.e. it enforces, at the label level, that nothing past
    `grid.n_steps` is ever read. The walk-forward split (src/splits.py, not
    yet written) must additionally leave a gap of >= H slots between the
    last training `t` and the first test `t`, for the same reason the
    organizers require "evaluate only on data after the forecast time": a
    training row's horizon window must not overlap a test row's input
    window. See tests/test_labels.py::test_build_label_frame_rejects_window_that_reads_past_grid.
    """
    if t_start_idx < 0:
        raise ValueError(f"t_start_idx must be >= 0, got {t_start_idx}")
    max_h = max(horizons.values())
    if t_end_idx + max_h - 1 > grid.n_steps:
        raise ValueError(
            f"[t_start_idx={t_start_idx}, t_end_idx={t_end_idx}) with max "
            f"horizon {max_h} slots would read past the grid "
            f"(n_steps={grid.n_steps}); shrink t_end_idx or extend the grid."
        )
    if t_end_idx <= t_start_idx:
        return pd.DataFrame(columns=["oblast", "t", "mode"] + [f"target_{h}" for h in horizons])

    t_idx = np.arange(t_start_idx, t_end_idx)
    active_now = grid.state[:, t_idx]  # [n_oblast, n_t]
    n_oblast, n_t = active_now.shape

    frame = pd.DataFrame(
        {
            "oblast": np.repeat(grid.oblasts, n_t),
            "t": np.tile(grid.t0 + t_idx * SLOT, n_oblast),
            "mode": np.where(active_now.ravel(order="C"), "continuation", "onset"),
        }
    )
    for h_name, h_slots in horizons.items():
        onset = (grid.onset_cum[:, t_idx + h_slots] - grid.onset_cum[:, t_idx]) > 0
        continuation = grid.state[:, t_idx + h_slots]
        target = np.where(active_now, continuation, onset)
        frame[f"target_{h_name}"] = target.ravel(order="C").astype(int)
    return frame
