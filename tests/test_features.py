"""Tests for src/features.py, written together with it.

The central one is the leakage test: for several cut points t_cut, rows with
t <= t_cut must be IDENTICAL whether events that start strictly after t_cut
are present, deleted, or moved days later, and whether the end of an alert
that is still running at t_cut is extended. Only raw timestamps <= t may
influence the row at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_io import union_intervals
from src.features import CAP_MIN, build_feature_frame, feature_columns
from src.labels import build_grid, build_label_frame

SLOT = pd.Timedelta("30min")
T0 = pd.Timestamp("2026-01-01 16:00:00", tz="UTC")  # slot 0; 18:00 Kyiv local time (EET, UTC+2)
EMPTY = pd.DataFrame(
    {
        "oblast": pd.Series([], dtype=object),
        "started_at": pd.to_datetime([], utc=True),
        "finished_at": pd.to_datetime([], utc=True),
    }
)


def _ep(oblast: str, start: str, end: str) -> dict:
    return {"oblast": oblast, "started_at": pd.Timestamp(start, tz="UTC"), "finished_at": pd.Timestamp(end, tz="UTC")}


def _frame(df, oblasts, adj, n_steps, t_start, t_end, t0=T0):
    grid = build_grid(df, t0, n_steps, oblasts=oblasts)
    return build_feature_frame(grid, union_intervals(df), adj, t_start, t_end)


def _row(frame, oblast, t_idx, t0=T0):
    return frame[(frame.oblast == oblast) & (frame.t == t0 + t_idx * SLOT)].iloc[0]


# --------------------------------------------------------------------------- #
# Hand-computed values on a tiny case
# --------------------------------------------------------------------------- #
def test_hand_computed_feature_values():
    df = pd.DataFrame(
        [
            _ep("A", "2026-01-01 16:05", "2026-01-01 16:25"),  # zero-slot (both -> 16:30)
            _ep("A", "2026-01-01 17:10", "2026-01-01 18:20"),  # slots 3,4 active (17:30, 18:00)
            _ep("B", "2026-01-01 17:20", "2026-01-01 17:40"),  # slot 3 active only
        ]
    )
    adj = {"A": ["B"], "B": ["A"], "C": []}
    f = _frame(df, ["A", "B", "C"], adj, n_steps=12, t_start=0, t_end=9)

    r = _row(f, "A", 4)  # 18:00
    assert r.active_now == 1
    assert (r.lag_active_1, r.lag_active_2, r.lag_active_4) == (1, 0, 0)   # 17:10 ceils to 17:30
    assert r.active_frac_4 == 0.5 and np.isclose(r.active_frac_12, 2 / 12)
    assert r.onset_cnt_4 == 2                                              # zero-slot blip + real alert
    assert np.isclose(r.mins_since_last_start, 50)                          # 17:10 -> 18:00
    assert np.isclose(r.mins_since_last_end, 95)                            # 16:25 -> 18:00
    assert np.isclose(r.active_for_min, 50)
    assert r.nb_active_cnt == 0 and np.isclose(r.nb_active_frac_12, 1 / 12)
    assert r.nb_onset_cnt_4 == 1 and r.nb_onset_cnt_12 == 1
    assert r.nat_active_cnt == 1 and r.nat_active_delta_2 == 1 and r.nat_onset_cnt_4 == 3
    assert (r.slot_of_day, r.dow) == (40, 3)                                # 20:00 local, Thursday

    r = _row(f, "A", 0)  # 16:00: nothing has started yet
    assert r.mins_since_last_start == CAP_MIN and r.mins_since_last_end == CAP_MIN
    assert (r.active_now, r.active_for_min) == (0, 0.0)
    assert (r.slot_of_day, r.dow) == (36, 3)

    r = _row(f, "A", 1)  # 16:30: the zero-slot alert is over; never "active" on the grid
    assert r.active_now == 0 and r.active_for_min == 0
    assert np.isclose(r.mins_since_last_start, 25) and np.isclose(r.mins_since_last_end, 5)

    r = _row(f, "C", 4)  # an oblast with no events at all is present and all-quiet
    assert r.active_now == 0 and r.mins_since_last_start == CAP_MIN and r.nb_active_cnt == 0


def test_event_starting_exactly_at_t_is_visible_at_t():
    df = pd.DataFrame([_ep("A", "2026-01-01 17:00", "2026-01-01 17:50")])
    f = _frame(df, ["A"], {"A": []}, n_steps=8, t_start=0, t_end=6)
    r = _row(f, "A", 2)  # 17:00, raw start == t
    assert r.active_now == 1 and r.mins_since_last_start == 0 and r.active_for_min == 0 and r.onset_cnt_4 == 1
    assert _row(f, "A", 1).active_now == 0  # 16:30: not yet


def test_calendar_features_use_kyiv_local_time_including_dst_repeat():
    # Fall-back on 2026-10-25: 04:00 EEST -> 03:00 EET, i.e. 01:00 UTC.
    t0 = pd.Timestamp("2026-10-25 00:00:00", tz="UTC")
    f = _frame(EMPTY, ["A"], {"A": []}, n_steps=6, t_start=0, t_end=5, t0=t0)
    # 00:00,00:30 UTC are 03:00,03:30 EEST; 01:00,01:30 UTC are 03:00,03:30 EET (repeated); 02:00 UTC = 04:00 EET.
    assert f.slot_of_day.tolist() == [6, 7, 6, 7, 8]
    summer = _frame(EMPTY, ["A"], {"A": []}, n_steps=2, t_start=0, t_end=1, t0=pd.Timestamp("2026-07-01 16:00", tz="UTC"))
    assert summer.slot_of_day.iloc[0] == 38  # 16:00 UTC = 19:00 EEST


def test_feature_and_label_frames_are_row_aligned():
    df = pd.DataFrame([_ep("A", "2026-01-01 17:00", "2026-01-01 17:50"), _ep("B", "2026-01-01 16:40", "2026-01-01 17:10")])
    grid = build_grid(df, T0, 20, oblasts=["A", "B"])
    feats = build_feature_frame(grid, union_intervals(df), {"A": ["B"], "B": ["A"]}, 3, 10)
    labels = build_label_frame(grid, 3, 10, horizons={"1h": 2})
    assert feats["oblast"].tolist() == labels["oblast"].tolist()
    assert (feats["t"].to_numpy() == labels["t"].to_numpy()).all()
    assert list(feats.columns) == ["oblast", "t"] + feature_columns()


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
OBLASTS = ["A", "B", "C", "D"]
ADJ = {"A": ["B", "C"], "B": ["A", "D"], "C": ["A"], "D": ["B"]}
N_DAYS = 3
N_STEPS = N_DAYS * 48 + 24


def _random_events(seed: int, n_per_oblast: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for o in OBLASTS:
        for _ in range(n_per_oblast):
            start = T0 + pd.Timedelta(minutes=int(rng.integers(0, N_DAYS * 24 * 60)))
            if rng.random() < 0.3:
                start = start.floor("30min")  # lots of exact slot-boundary starts
            end = start + pd.Timedelta(minutes=int(rng.integers(0, 180)))  # includes 0-min and zero-slot alerts
            rows.append({"oblast": o, "started_at": start, "finished_at": end})
    return pd.DataFrame(rows)


def _perturb(df: pd.DataFrame, t_cut: pd.Timestamp, how: str) -> pd.DataFrame:
    future = df["started_at"] > t_cut
    if how == "delete_future":
        return df[~future].reset_index(drop=True)
    d = df.copy()
    if how == "move_future_3_days_later":
        d.loc[future, ["started_at", "finished_at"]] += pd.Timedelta(days=3)
    elif how == "move_future_to_just_after_cut":
        # keep their durations, but squeeze them right after t_cut (maximally adversarial for any lookahead)
        shift = (d.loc[future, "started_at"] - t_cut) - pd.Timedelta(minutes=1)
        d.loc[future, "started_at"] -= shift
        d.loc[future, "finished_at"] -= shift
    elif how == "extend_end_of_ongoing_alerts":
        ongoing = (d["started_at"] <= t_cut) & (d["finished_at"] > t_cut)
        d.loc[ongoing, "finished_at"] += pd.Timedelta(hours=5)
    else:
        raise ValueError(how)
    return d


def test_leakage_feature_rows_at_or_before_t_ignore_everything_after_t():
    n_checked = 0
    for seed in (0, 1, 2):
        df = _random_events(seed)
        for t_cut_idx in (30, 61, 100):
            t_cut = T0 + t_cut_idx * SLOT
            base = _frame(df, OBLASTS, ADJ, N_STEPS, 0, t_cut_idx + 1)
            for how in (
                "delete_future",
                "move_future_3_days_later",
                "move_future_to_just_after_cut",
                "extend_end_of_ongoing_alerts",
            ):
                perturbed = _frame(_perturb(df, t_cut, how), OBLASTS, ADJ, N_STEPS, 0, t_cut_idx + 1)
                pd.testing.assert_frame_equal(base, perturbed, obj=f"seed={seed} t_cut={t_cut_idx} {how}")
                n_checked += 1
    assert n_checked == 3 * 3 * 4


def test_leakage_test_is_not_vacuous_it_detects_a_feature_that_peeks():
    """Sanity check of the test itself: events that start BEFORE t_cut must
    change the rows at/before t_cut (otherwise the equality above proves nothing)."""
    df = _random_events(0)
    t_cut_idx = 61
    t_cut = T0 + t_cut_idx * SLOT
    base = _frame(df, OBLASTS, ADJ, N_STEPS, 0, t_cut_idx + 1)
    past = df["started_at"] <= t_cut
    changed = df.copy()
    changed.loc[past, ["started_at", "finished_at"]] += pd.Timedelta(days=3)  # remove the past instead
    other = _frame(changed, OBLASTS, ADJ, N_STEPS, 0, t_cut_idx + 1)
    assert not base.equals(other)


def test_grid_state_matches_the_raw_definition_start_le_t_lt_end():
    df = _random_events(5)
    grid = build_grid(df, T0, N_STEPS, oblasts=OBLASTS)
    eps = union_intervals(df)
    for o_i, o in enumerate(OBLASTS):
        e = eps[eps.oblast == o]
        for i in range(0, N_STEPS + 1):
            t = T0 + i * SLOT
            raw_active = bool(((e.started_at <= t) & (e.finished_at > t)).any())
            assert bool(grid.state[o_i, i]) == raw_active, (o, i)


def test_leakage_check_catches_a_deliberately_forward_looking_feature():
    """Mutation test of the leakage test: replace the trailing-window helper by
    one that sums the NEXT w slots (t .. t+w-1). The comparison used above must
    then see rows at/before t_cut change when future events are deleted."""
    import src.features as features_module

    original = features_module._trailing_sum

    def peeking(x, w):
        out = np.zeros(x.shape, dtype=np.float64)
        for k in range(min(w, x.shape[1])):
            out[:, : x.shape[1] - k] += x[:, k:]
        return out

    features_module._trailing_sum = peeking
    try:
        df = _random_events(0)
        t_cut_idx = 61
        t_cut = T0 + t_cut_idx * SLOT
        base = _frame(df, OBLASTS, ADJ, N_STEPS, 0, t_cut_idx + 1)
        deleted = _frame(_perturb(df, t_cut, "delete_future"), OBLASTS, ADJ, N_STEPS, 0, t_cut_idx + 1)
        assert not base.equals(deleted), "a forward-looking feature went undetected: the leakage test is vacuous"
    finally:
        features_module._trailing_sum = original
