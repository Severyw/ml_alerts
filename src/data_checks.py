"""Data-quality checks run before any feature engineering.

Run as:
    python -m src.data_checks --data-dir data/raw

Prints, for both the official and volunteer files:
  1. Freshness: last record date, and days since it, for each file.
  2. Duplicate rows in the official file (a known ingestion artifact).
  3. Zero-slot share: the fraction of oblast-level "in alert" episodes that
     collapse to zero 30-minute slots once start/end are ceil'd to the grid
     (per the agreed ceil-rounding convention).
  4. Positive rates for the two label definitions (onset, continuation)
     at both horizons (1h, 6h), computed on three different time windows,
     so the reported distribution shift between windows is visible rather
     than asserted.
  5. A heuristic, NOT a measurement, of how much the official file's 3-hour
     repeat-activation rule can shift `started_at` later than the true
     start. This cannot be measured exactly from the processed CSV alone
     (see docstring of `estimate_started_at_shift`), so the number below is
     a lower-bound proxy with a stated confound, not a verified figure.
"""
from __future__ import annotations

import argparse
import itertools
import pathlib

import numpy as np
import pandas as pd

from src.data_io import drop_excluded_oblasts, load_raw, union_intervals

SLOT = pd.Timedelta("30min")


# --------------------------------------------------------------------------- #
# 1. Freshness
# --------------------------------------------------------------------------- #
def report_freshness(raw) -> None:
    print("=" * 70)
    print("1. FRESHNESS")
    print("=" * 70)
    now = pd.Timestamp.now(tz="UTC")
    for name, df, path in (
        ("official", raw.official, raw.official_path),
        ("volunteer", raw.volunteer, raw.volunteer_path),
    ):
        last_start = df.started_at.max()
        last_finish = df.finished_at.max()
        age = now - last_finish
        print(f"[{name}] file: {path.name}")
        print(f"  rows: {len(df):,}")
        print(f"  last started_at:  {last_start}")
        print(f"  last finished_at: {last_finish}")
        print(f"  age of last record vs now: {age}")
        print()


# --------------------------------------------------------------------------- #
# 2. Duplicates
# --------------------------------------------------------------------------- #
def report_duplicates(data_dir: pathlib.Path) -> None:
    print("=" * 70)
    print("2. DUPLICATE ROWS (raw file, before dedup in data_io.load_raw)")
    print("=" * 70)
    off = pd.read_csv(data_dir / "official_data_en.csv")
    vol = pd.read_csv(data_dir / "volunteer_data_en.csv")
    print(f"official: {len(off):,} rows, {off.duplicated().sum():,} exact "
          f"duplicate rows ({off.duplicated().sum() / len(off):.2%})")
    print(f"volunteer: {len(vol):,} rows, {vol.duplicated().sum():,} exact "
          f"duplicate rows ({vol.duplicated().sum() / len(vol):.2%})")
    print()


# --------------------------------------------------------------------------- #
# 3. Zero-slot share after ceil-to-30min
# --------------------------------------------------------------------------- #
def ceil_slot_count(episodes: pd.DataFrame) -> pd.Series:
    start = episodes.started_at.dt.ceil(SLOT)
    end = episodes.finished_at.dt.ceil(SLOT)
    return ((end - start) / SLOT).astype(int)


def report_zero_slot(raw) -> None:
    print("=" * 70)
    print("3. ZERO-SLOT SHARE (oblast-level episodes, ceil to 30-min grid)")
    print("=" * 70)
    for name, df in (("official", raw.official), ("volunteer", raw.volunteer)):
        df = drop_excluded_oblasts(df)
        episodes = union_intervals(df)
        n_slots = ceil_slot_count(episodes)
        dur_min = (episodes.finished_at - episodes.started_at).dt.total_seconds() / 60
        print(f"[{name}] episodes: {len(episodes):,}")
        print(f"  zero-slot share: {(n_slots == 0).mean():.2%}")
        print(f"  <=1 slot share:  {(n_slots <= 1).mean():.2%}")
        print(f"  median duration: {dur_min.median():.1f} min")
        by_year = (n_slots == 0).groupby(episodes.started_at.dt.year).mean()
        print(f"  zero-slot share by year: {by_year.round(4).to_dict()}")
        print()


# --------------------------------------------------------------------------- #
# 4. Positive rates per label / horizon / window
# --------------------------------------------------------------------------- #
def build_state_grid(df: pd.DataFrame, t0: pd.Timestamp, n_steps: int):
    """Boolean [n_oblasts, n_steps+1] grid: state[o, i] = oblast o has an
    active alert at slot boundary t0 + i*30min (state known strictly at or
    before that boundary, consistent with ceil-rounding)."""
    episodes = union_intervals(df)
    oblasts = sorted(episodes.oblast.unique())
    idx = {o: i for i, o in enumerate(oblasts)}
    delta = np.zeros((len(oblasts), n_steps + 2), dtype=np.int32)
    onset = np.zeros((len(oblasts), n_steps + 2), dtype=np.int32)
    for o, s, e in zip(episodes.oblast, episodes.started_at, episodes.finished_at):
        i0 = int(np.ceil((s - t0) / SLOT))
        i1 = int(np.ceil((e - t0) / SLOT))
        if i0 < 0 or i0 > n_steps:
            continue
        i1 = min(i1, n_steps + 1)
        onset[idx[o], i0] += 1
        if i1 > i0:
            delta[idx[o], i0] += 1
            if i1 <= n_steps:
                delta[idx[o], i1] -= 1
    state = np.cumsum(delta, axis=1) > 0
    onset_cum = np.cumsum(onset, axis=1)
    return state, onset_cum, oblasts


def positive_rates(df: pd.DataFrame, lo: str, hi: str) -> dict:
    """Onset = a new alert starts in (t, t+H]; computed only on rows with no
    alert active at t. Continuation = the alert still active at exactly
    t+H (not 'active at any point in the window' -- see chat discussion of
    why that definition was trivial)."""
    t0 = pd.Timestamp(lo, tz="UTC")
    t1 = pd.Timestamp(hi, tz="UTC")
    n_steps = int((t1 - t0) / SLOT) + 12  # pad so t+6h lookups near the edge are valid
    state, onset_cum, _ = build_state_grid(df, t0, n_steps)
    a, b = 0, int((t1 - t0) / SLOT)
    ts = np.arange(a, b)
    active_now = state[:, ts]
    out = {"n_rows_no_alert": int((~active_now).sum()), "n_rows_alert": int(active_now.sum())}
    for horizon_slots, tag in ((2, "1h"), (12, "6h")):
        new_onset = (onset_cum[:, ts + horizon_slots] - onset_cum[:, ts]) > 0
        still_active = state[:, ts + horizon_slots]
        out[f"onset_{tag}"] = round(new_onset[~active_now].mean(), 4)
        out[f"continuation_{tag}"] = round(still_active[active_now].mean(), 4)
    return out


def report_positive_rates(raw) -> None:
    print("=" * 70)
    print("4. POSITIVE RATES (onset vs continuation, 1h vs 6h, by window)")
    print("   onset: no alert at t -> new alert starts in (t, t+H]")
    print("   continuation: alert active at t -> still active at exactly t+H")
    print("=" * 70)
    windows = [
        ("2024-01-01", "2025-11-30", "pre-raion-switch era"),
        ("2025-12-15", "2026-08-28", "raion-level era"),
    ]
    for name, df in (("official", raw.official), ("volunteer", raw.volunteer)):
        df = drop_excluded_oblasts(df)
        print(f"[{name}]")
        for lo, hi, label in windows:
            if df.started_at.max() < pd.Timestamp(lo, tz="UTC"):
                continue
            rates = positive_rates(df, lo, hi)
            print(f"  {lo}..{hi} ({label}): {rates}")
        # Most recent 5 weeks available in THIS file specifically
        hi = (df.finished_at.max() - pd.Timedelta("12h")).floor("D")
        lo = hi - pd.Timedelta(days=36)
        if lo > pd.Timestamp("2022-03-15", tz="UTC"):
            rates = positive_rates(df, str(lo.date()), str(hi.date()))
            print(f"  {lo.date()}..{hi.date()} (last 5 weeks in this file): {rates}")
        print()


# --------------------------------------------------------------------------- #
# 5. started_at shift heuristic
# --------------------------------------------------------------------------- #
def estimate_started_at_shift(raw, lo: str = "2023-01-01", hi: str = "2025-08-31") -> None:
    """Heuristic proxy for the 3-hour repeat-activation rule in both
    processors (processors/official_channel_processor.py and
    processors/volunteer_etryvoga_processor.py): if a second "activated"
    message for the same location arrives while an alert is already open
    and within 3 hours of its logged started_at, the open alert is silently
    replaced in memory by a new one with the LATER started_at -- no
    completed row is ever written for the original, so the discarded start
    time leaves no trace in the CSV we can query.

    We cannot measure this exactly without the raw Telegram message history
    (not available here: it requires Telegram API credentials we don't
    have). As a proxy, we match each official episode to the volunteer
    episode in the same oblast whose end time is closest (within 20
    minutes), on the assumption that a matched pair describes the same
    real-world siren event. We then look at started_at_official -
    started_at_volunteer for matched pairs:
      - A broad spread of small positive deltas is CONSISTENT with the bug
        (and also with ordinary cross-channel reporting lag -- we cannot
        separate the two from this data).
      - Deltas beyond 3 hours are NOT this bug: they fall under the "close
        and start a new episode" branch, which does not shift anything.
    This is reported as a bound, not a verified count.
    """
    print("=" * 70)
    print("5. started_at SHIFT FROM 3H REPEAT-ACTIVATION RULE (heuristic proxy)")
    print("=" * 70)
    off = union_intervals(drop_excluded_oblasts(raw.official))
    vol = union_intervals(drop_excluded_oblasts(raw.volunteer))
    off = off[(off.started_at >= lo) & (off.started_at < hi)]
    vol = vol[(vol.started_at >= lo) & (vol.started_at < hi)]

    def to_ns(series: pd.Series) -> np.ndarray:
        # tz-aware Series -> naive datetime64[ns] int64 nanoseconds, safe for
        # subtraction (we only ever compare timestamps within this function).
        return series.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype("int64")

    deltas = []
    for oblast, og in off.groupby("oblast"):
        vg = vol[vol.oblast == oblast]
        if vg.empty:
            continue
        v_end = to_ns(vg.finished_at)
        v_start = to_ns(vg.started_at)
        o_start = to_ns(og.started_at)
        o_end = to_ns(og.finished_at)
        for s_o, e_o in zip(o_start, o_end):
            j = np.argmin(np.abs(v_end - e_o))
            if abs((v_end[j] - e_o) / 60e9) <= 20:
                delta_min = (s_o - v_start[j]) / 60e9
                deltas.append(delta_min)
    deltas = np.array(deltas)
    within_3h = deltas[(deltas > 0) & (deltas <= 180)]
    print(f"matched pairs (same oblast, ends within 20 min of each other): {len(deltas):,}")
    if len(deltas):
        print(f"  share with official started_at LATER than volunteer's, by <=3h: "
              f"{len(within_3h) / len(deltas):.2%}")
        print(f"  among those, median delta: {np.median(within_3h):.1f} min, "
              f"90th pct: {np.percentile(within_3h, 90):.1f} min" if len(within_3h) else "  (none)")
        print(f"  share with official started_at LATER by >3h "
              f"(a new episode branch, not this bug): {(deltas > 180).mean():.2%}")
        print(f"  share with official started_at EARLIER (delta < 0): {(deltas < 0).mean():.2%}")
    print("CAVEAT: this conflates the ingestion bug with ordinary cross-channel\n"
          "reporting lag between the two Telegram sources; treat the numbers\n"
          "above as an upper bound on how often onset timing could be off by\n"
          "a few hours, not a confirmed defect count. We did not have Telegram\n"
          "API credentials to re-derive this from the raw messages.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()
    data_dir = pathlib.Path(args.data_dir)

    raw = load_raw(data_dir)
    report_freshness(raw)
    report_duplicates(data_dir)
    report_zero_slot(raw)
    report_positive_rates(raw)
    estimate_started_at_shift(raw)


if __name__ == "__main__":
    main()
