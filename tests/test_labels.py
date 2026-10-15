"""Tests for src/labels.py, written together with it (not after).

Covers: ceil-boundary edge cases (exact / +1 min / -1 min), zero-slot
alerts, the horizon gap at the right edge of a grid, and a leakage check
that the *input* side (state/onset_cum at or before t) cannot be changed
by deleting or shuffling events that happen strictly after t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels import build_grid, build_label_frame

SLOT = pd.Timedelta("30min")
T0 = pd.Timestamp("2026-01-01 16:00:00", tz="UTC")  # slot index 0


def _episode(oblast: str, start: str, end: str) -> dict:
    return {
        "oblast": oblast,
        "started_at": pd.Timestamp(start, tz="UTC"),
        "finished_at": pd.Timestamp(end, tz="UTC"),
    }


def _df(*episodes: dict) -> pd.DataFrame:
    return pd.DataFrame(list(episodes))


# --------------------------------------------------------------------------- #
# Ceil-boundary edge cases
# --------------------------------------------------------------------------- #
def test_ceil_boundary_exact_start_on_slot_boundary():
    # Starts exactly at 16:30 (slot 1): visible from slot 1, not slot 0.
    df = _df(_episode("A", "2026-01-01 16:30:00", "2026-01-01 17:30:00"))
    grid = build_grid(df, T0, n_steps=4)
    assert grid.state[0, 0] is np.bool_(False)  # 16:00, before the alert
    assert grid.state[0, 1] is np.bool_(True)   # 16:30, exactly at start
    assert grid.state[0, 2] is np.bool_(True)   # 17:00, still active
    # finished_at=17:30 ceils to slot 3 (its own boundary). The grid is
    # half-open [start_slot, end_slot): slot 3 is the first FALSE slot, i.e.
    # the alert is NOT active exactly at its own (ceil'd) end slot.
    assert not grid.state[0, 3]


def test_ceil_boundary_plus_one_minute_rounds_up_to_next_slot():
    # Starts at 16:31 -> ceil -> 17:00 (slot 2), NOT 16:30 (slot 1).
    df = _df(_episode("A", "2026-01-01 16:31:00", "2026-01-01 17:30:00"))
    grid = build_grid(df, T0, n_steps=4)
    assert not grid.state[0, 1]  # 16:30: must NOT be active yet
    assert grid.state[0, 2]      # 17:00: now active


def test_ceil_boundary_minus_one_minute_rounds_up_to_same_slot():
    # Starts at 16:29 -> ceil -> 16:30 (slot 1), same as exactly-16:30.
    df = _df(_episode("A", "2026-01-01 16:29:00", "2026-01-01 17:30:00"))
    grid = build_grid(df, T0, n_steps=4)
    assert not grid.state[0, 0]  # 16:00: before
    assert grid.state[0, 1]      # 16:30: active (16:29 ceils here)


# --------------------------------------------------------------------------- #
# Zero-slot alerts
# --------------------------------------------------------------------------- #
def test_zero_slot_alert_counts_as_onset_but_never_active():
    # 16:05-16:25: both start and end ceil to 16:30 (slot 1) -> zero-slot.
    df = _df(_episode("A", "2026-01-01 16:05:00", "2026-01-01 16:25:00"))
    grid = build_grid(df, T0, n_steps=4)
    # Never appears as "active" anywhere on the grid.
    assert not grid.state[0, :].any()
    # But it DID happen: onset_cum must step up by 1 at slot 1.
    assert grid.onset_cum[0, 0] == 0
    assert grid.onset_cum[0, 1] == 1
    assert grid.onset_cum[0, 4] == 1

    # And it must still show up as a positive onset label for a row at t=slot 0.
    frame = build_label_frame(grid, t_start_idx=0, t_end_idx=1, horizons={"1h": 2})
    row = frame.iloc[0]
    assert row["mode"] == "onset"       # nothing active at t=slot 0
    assert row["target_1h"] == 1        # but the zero-slot alert onsets within 1h


def test_zero_slot_alert_mixed_with_normal_alert_in_same_oblast():
    # A zero-slot blip at 16:05-16:25 (slot 1), then a real alert 17:00-18:00.
    df = _df(
        _episode("A", "2026-01-01 16:05:00", "2026-01-01 16:25:00"),
        _episode("A", "2026-01-01 17:00:00", "2026-01-01 18:00:00"),
    )
    grid = build_grid(df, T0, n_steps=6)
    assert grid.onset_cum[0, -1] == 2  # both onsets counted
    assert not grid.state[0, 1]        # the zero-slot one never shows active
    assert grid.state[0, 2]            # 17:00 slot: the real alert


# --------------------------------------------------------------------------- #
# Onset / continuation / mode correctness
# --------------------------------------------------------------------------- #
def test_continuation_is_active_at_exactly_t_plus_h_not_any_time_in_window():
    # Alert active 16:00-16:45 only (ends before slot 2 = 17:00).
    df = _df(_episode("A", "2026-01-01 16:00:00", "2026-01-01 16:45:00"))
    grid = build_grid(df, T0, n_steps=4)
    # At t=slot 0 (16:00) the alert IS active, so mode="continuation".
    frame = build_label_frame(grid, t_start_idx=0, t_end_idx=1, horizons={"1h": 2})
    row = frame.iloc[0]
    assert row["mode"] == "continuation"
    # It is NOT active at exactly t+1h (slot 2 = 17:00), even though it WAS
    # active at some point within (t, t+1h] (namely slot 1 = 16:30). This is
    # exactly the distinction the "any active in window" definition blurred.
    assert row["target_1h"] == 0


def test_onset_window_is_left_open_right_closed():
    # Onset at slot 4 (18:00-18:30); window (t, t+H] is right-CLOSED, so an
    # onset landing exactly on the t+H boundary must count.
    df = _df(_episode("A", "2026-01-01 18:00:00", "2026-01-01 18:30:00"))
    grid = build_grid(df, T0, n_steps=8)
    frame = build_label_frame(grid, t_start_idx=1, t_end_idx=5, horizons={"1h": 2})
    by_t = frame.set_index(frame["t"])

    t1, t2, t4 = T0 + SLOT, T0 + 2 * SLOT, T0 + 4 * SLOT  # 16:30, 17:00, 18:00
    assert by_t.loc[t1, "mode"] == "onset"
    assert by_t.loc[t1, "target_1h"] == 0   # (16:30, 17:30] does not reach the 18:00 onset
    assert by_t.loc[t2, "mode"] == "onset"
    assert by_t.loc[t2, "target_1h"] == 1   # (17:00, 18:00] -- 18:00 is the closed right edge
    assert by_t.loc[t4, "mode"] == "continuation"  # active exactly at 18:00
    assert by_t.loc[t4, "target_1h"] == 0           # but no longer active at 19:00 (ended 18:30)


# --------------------------------------------------------------------------- #
# Horizon gap at the right edge
# --------------------------------------------------------------------------- #
def test_build_label_frame_rejects_window_that_reads_past_grid():
    df = _df(_episode("A", "2026-01-01 16:00:00", "2026-01-01 16:30:00"))
    grid = build_grid(df, T0, n_steps=5)
    # Largest valid t_end_idx for horizon=2 slots is n_steps - 2 + 1 = 4.
    build_label_frame(grid, 0, 4, horizons={"1h": 2})  # must not raise
    with pytest.raises(ValueError):
        build_label_frame(grid, 0, 5, horizons={"1h": 2})  # t_idx=4 -> needs slot 6 > n_steps=5


def test_build_label_frame_empty_window_returns_empty_frame_not_error():
    df = _df(_episode("A", "2026-01-01 16:00:00", "2026-01-01 16:30:00"))
    grid = build_grid(df, T0, n_steps=5)
    frame = build_label_frame(grid, 3, 3, horizons={"1h": 2})
    assert len(frame) == 0


# --------------------------------------------------------------------------- #
# Leakage: the INPUT side (state/onset_cum at/before t) must be invariant to
# events that happen strictly after t, however they are perturbed.
# --------------------------------------------------------------------------- #
def test_leakage_state_at_t_is_invariant_to_future_events_being_deleted():
    base = [
        _episode("A", "2026-01-01 16:00:00", "2026-01-01 16:30:00"),
        _episode("A", "2026-01-01 20:00:00", "2026-01-01 21:00:00"),  # "future" relative to t below
        _episode("B", "2026-01-01 17:00:00", "2026-01-01 17:30:00"),
    ]
    t_cutoff_idx = 2  # 17:00 -- only the first A-episode and the B-episode start at/before this
    with_future = build_grid(_df(*base), T0, n_steps=12)
    without_future = build_grid(_df(*base[:1], base[2]), T0, n_steps=12)  # future A-episode deleted

    np.testing.assert_array_equal(
        with_future.state[:, : t_cutoff_idx + 1], without_future.state[:, : t_cutoff_idx + 1]
    )
    np.testing.assert_array_equal(
        with_future.onset_cum[:, : t_cutoff_idx + 1], without_future.onset_cum[:, : t_cutoff_idx + 1]
    )


def test_leakage_state_at_t_is_invariant_to_future_events_being_shuffled():
    base = [
        _episode("A", "2026-01-01 16:00:00", "2026-01-01 16:30:00"),
        _episode("A", "2026-01-01 20:00:00", "2026-01-01 21:00:00"),
        _episode("B", "2026-01-01 17:00:00", "2026-01-01 17:30:00"),
    ]
    t_cutoff_idx = 2  # 17:00
    moved = list(base)
    moved[1] = _episode("A", "2026-01-02 09:00:00", "2026-01-02 10:00:00")  # same event, later time

    g1 = build_grid(_df(*base), T0, n_steps=100)
    g2 = build_grid(_df(*moved), T0, n_steps=100)

    np.testing.assert_array_equal(g1.state[:, : t_cutoff_idx + 1], g2.state[:, : t_cutoff_idx + 1])
    np.testing.assert_array_equal(
        g1.onset_cum[:, : t_cutoff_idx + 1], g2.onset_cum[:, : t_cutoff_idx + 1]
    )


def test_leakage_label_frame_up_to_t_unaffected_by_perturbing_a_future_event():
    """End-to-end version of the leakage check at the build_label_frame level:
    rows with t <= t_cutoff must be byte-for-byte identical whether or not a
    strictly-later event exists, is deleted, or is moved further still."""
    def make_frame(episodes):
        grid = build_grid(_df(*episodes), T0, n_steps=20)
        return build_label_frame(grid, 0, 5, horizons={"1h": 2})  # rows t=slot 0..4 (16:00..17:30)

    with_future = [
        _episode("A", "2026-01-01 16:00:00", "2026-01-01 16:15:00"),
        _episode("A", "2026-01-02 12:00:00", "2026-01-02 13:00:00"),  # well after row window
    ]
    without_future = with_future[:1]
    shuffled_future = [with_future[0], _episode("A", "2026-01-03 00:00:00", "2026-01-03 01:00:00")]

    f1 = make_frame(with_future)
    f2 = make_frame(without_future)
    f3 = make_frame(shuffled_future)
    pd.testing.assert_frame_equal(f1, f2)
    pd.testing.assert_frame_equal(f1, f3)


# --------------------------------------------------------------------------- #
# An alert whose raw start is exactly t belongs to the state AT t; it is not
# an onset "in (t, t+H]".
# --------------------------------------------------------------------------- #
def test_alert_starting_exactly_at_t_is_state_at_t_not_onset_in_window():
    df = _df(_episode("A", "2026-01-01 16:30:00", "2026-01-01 17:30:00"))  # raw start == slot 1
    grid = build_grid(df, T0, n_steps=6)

    # t = slot 1 (16:30): the alert is already part of the state ...
    assert grid.state[0, 1]
    # ... and there is NO onset in (t, t+1h] = slots 2..3:
    assert grid.onset_cum[0, 1 + 2] - grid.onset_cum[0, 1] == 0
    row = build_label_frame(grid, 1, 2, horizons={"1h": 2}).iloc[0]
    assert row["mode"] == "continuation"   # so this row is a continuation row, never an onset row

    # One slot earlier (t = 16:00) the very same alert IS an onset in (t, t+1h]:
    assert grid.onset_cum[0, 0 + 2] - grid.onset_cum[0, 0] == 1
    row0 = build_label_frame(grid, 0, 1, horizons={"1h": 2}).iloc[0]
    assert row0["mode"] == "onset" and row0["target_1h"] == 1


def test_zero_duration_alert_starting_exactly_at_t_is_not_an_onset_in_window():
    # start == end == 16:30: zero-slot, counted as an onset AT slot 1, hence not in (slot 1, slot 3].
    df = _df(_episode("A", "2026-01-01 16:30:00", "2026-01-01 16:30:00"))
    grid = build_grid(df, T0, n_steps=6)
    assert grid.onset_cum[0, 1] == 1
    assert grid.onset_cum[0, 3] - grid.onset_cum[0, 1] == 0


# --------------------------------------------------------------------------- #
# Fixed oblast universe
# --------------------------------------------------------------------------- #
def test_explicit_oblast_universe_keeps_rows_for_oblasts_without_events():
    df = _df(_episode("A", "2026-01-01 16:30:00", "2026-01-01 17:30:00"))
    grid = build_grid(df, T0, n_steps=6, oblasts=["A", "B", "C"])
    assert grid.oblasts == ["A", "B", "C"]
    assert grid.state.shape[0] == 3
    assert not grid.state[1].any() and not grid.state[2].any()
    assert grid.onset_cum[1].max() == 0


def test_events_outside_declared_universe_raise():
    df = _df(_episode("Z", "2026-01-01 16:30:00", "2026-01-01 17:30:00"))
    with pytest.raises(ValueError):
        build_grid(df, T0, n_steps=6, oblasts=["A", "B"])
