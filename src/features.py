"""Feature rows for (oblast, t). Every feature at slot t uses only information
with a raw timestamp <= t.

How that is enforced (and tested in tests/test_features.py):
* Grid-based features read `grid.state[:, :t+1]` / `grid.onset_cum[:, :t+1]`
  through causal operations only: lags (value at t-k), trailing windows
  (cumulative-sum differences ending at t), and cross-oblast sums at the
  same t. Nothing is ever indexed at t+1 or later.
* Raw-timestamp features use `numpy.searchsorted` on episode starts
  (side="right", i.e. start <= t) and on episode ends (end <= t). The end of
  an episode that is still running at t is never read.
* Calendar features are a function of t alone.
* The `naive` flag of the volunteer file is deliberately not used anywhere:
  it is only known after the fact (set when a later activation arrives).

COLD START. Windows that reach before the start of the grid are truncated
(missing history counts as "no alert"). Start modelling rows at
t_start_idx >= max(WINDOWS) (7 days) if you want full windows.

The oblast identity is NOT encoded here: the frame carries the `oblast`
string and the model code decides how to encode it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.config import SLOT
from src.labels import AlertGrid

KYIV_TZ = "Europe/Kyiv"
LAGS = (1, 2, 4, 12)          # slots: 30 min, 1 h, 2 h, 6 h
WINDOWS = (4, 12, 48, 336)    # slots: 2 h, 6 h, 24 h, 7 d
# Value used when a "minutes since ..." quantity is undefined because no such
# event exists yet. A constant (7 days), not fitted on any data.
CAP_MIN = 7 * 24 * 60

_NS_PER_MIN = 60_000_000_000
_SLOT_NS = int(SLOT // pd.Timedelta(1, "ns"))


def feature_columns() -> list[str]:
    cols = ["active_now"]
    cols += [f"lag_active_{k}" for k in LAGS]
    cols += [f"active_frac_{w}" for w in WINDOWS]
    cols += [f"onset_cnt_{w}" for w in WINDOWS]
    cols += ["mins_since_last_start", "mins_since_last_end", "active_for_min"]
    cols += ["nb_active_cnt", "nb_active_frac_12", "nb_onset_cnt_4", "nb_onset_cnt_12"]
    cols += ["nat_active_cnt", "nat_active_delta_2", "nat_onset_cnt_4"]
    cols += ["slot_of_day", "dow", "slot_sin", "slot_cos"]
    return cols


# --------------------------------------------------------------------------- #
# causal helpers on [n_oblast, T] arrays
# --------------------------------------------------------------------------- #
def _lag(x: np.ndarray, k: int) -> np.ndarray:
    """out[:, t] = x[:, t-k]; zeros where t-k < 0."""
    out = np.zeros_like(x)
    out[:, k:] = x[:, :-k]
    return out


def _trailing_sum(x: np.ndarray, w: int) -> np.ndarray:
    """out[:, t] = sum of x[:, t-w+1 .. t] (truncated at the grid start)."""
    cs = np.cumsum(x, axis=1)
    out = cs.copy()
    out[:, w:] = cs[:, w:] - cs[:, :-w]
    return out


def _to_ns(series: pd.Series) -> np.ndarray:
    return series.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype("int64")


def _neighbour_matrix(grid: AlertGrid, adjacency: Mapping[str, Sequence[str]]) -> np.ndarray:
    idx = {o: i for i, o in enumerate(grid.oblasts)}
    m = np.zeros((len(idx), len(idx)), dtype=np.float64)
    for o, nbrs in adjacency.items():
        if o not in idx:
            continue  # adjacency may list more oblasts than this grid's universe
        for n in nbrs:
            if n not in idx:
                raise ValueError(f"adjacency lists neighbour {n!r} of {o!r}, which is not in the grid universe")
            m[idx[o], idx[n]] = 1.0
    if not np.array_equal(m, m.T):
        raise ValueError("adjacency is not symmetric")
    return m


# --------------------------------------------------------------------------- #
def build_feature_frame(
    grid: AlertGrid,
    episodes: pd.DataFrame,
    adjacency: Mapping[str, Sequence[str]],
    t_start_idx: int,
    t_end_idx: int,
) -> pd.DataFrame:
    """One row per (oblast, t), t in slot indices [t_start_idx, t_end_idx).

    Row order is oblast-major, then t -- identical to
    src.labels.build_label_frame, so frames can be aligned directly.

    `episodes` must be the union_intervals() output for the same filtered
    data the grid was built from (columns: oblast, started_at, finished_at).
    """
    if t_start_idx < 0 or t_end_idx > grid.n_steps + 1:
        raise ValueError(f"[{t_start_idx}, {t_end_idx}) outside grid 0..{grid.n_steps}")
    if t_end_idx <= t_start_idx:
        return pd.DataFrame(columns=["oblast", "t"] + feature_columns())

    n_o = len(grid.oblasts)
    T = grid.n_steps + 1
    state = grid.state[:, :T].astype(np.int32)                 # [O, T]
    onset = np.diff(grid.onset_cum[:, :T], axis=1, prepend=0)  # [O, T] onsets starting in slot t
    cols: dict[str, np.ndarray] = {}

    # ---- own state, lags, trailing windows (all causal) ---------------------
    cols["active_now"] = state
    for k in LAGS:
        cols[f"lag_active_{k}"] = _lag(state, k)
    for w in WINDOWS:
        cols[f"active_frac_{w}"] = _trailing_sum(state, w) / w
        cols[f"onset_cnt_{w}"] = _trailing_sum(onset, w)

    # ---- raw-timestamp event features ------------------------------------
    t_idx = np.arange(t_start_idx, t_end_idx)
    t0_ns = grid.t0.as_unit("ns").value
    tq = t0_ns + t_idx.astype(np.int64) * _SLOT_NS
    since_start = np.full((n_o, len(t_idx)), float(CAP_MIN))
    since_end = np.full((n_o, len(t_idx)), float(CAP_MIN))
    for o_i, o in enumerate(grid.oblasts):
        ep = episodes[episodes["oblast"] == o].sort_values("started_at")
        if ep.empty:
            continue
        starts, ends = _to_ns(ep["started_at"]), _to_ns(ep["finished_at"])
        k = np.searchsorted(starts, tq, side="right") - 1   # last episode with start <= t
        has = k >= 0
        since_start[o_i, has] = (tq[has] - starts[k[has]]) / _NS_PER_MIN
        j = np.searchsorted(ends, tq, side="right") - 1     # last episode with end <= t
        has = j >= 0
        since_end[o_i, has] = (tq[has] - ends[j[has]]) / _NS_PER_MIN
    since_start = np.minimum(since_start, CAP_MIN)
    since_end = np.minimum(since_end, CAP_MIN)

    # ---- neighbours and national picture (same-t or trailing only) --------
    m = _neighbour_matrix(grid, adjacency)
    deg = np.maximum(m.sum(axis=1, keepdims=True), 1.0)
    cols["nb_active_cnt"] = m @ state
    cols["nb_active_frac_12"] = (m @ cols["active_frac_12"]) / deg
    cols["nb_onset_cnt_4"] = m @ cols["onset_cnt_4"]
    cols["nb_onset_cnt_12"] = m @ cols["onset_cnt_12"]
    nat_active = state.sum(axis=0, keepdims=True)              # [1, T]
    cols["nat_active_cnt"] = np.repeat(nat_active, n_o, axis=0)
    cols["nat_active_delta_2"] = np.repeat(nat_active - _lag(nat_active, 2), n_o, axis=0)
    cols["nat_onset_cnt_4"] = np.repeat(_trailing_sum(onset, 4).sum(axis=0, keepdims=True), n_o, axis=0)

    # ---- slice to requested rows, add the per-row quantities ---------------
    sl = slice(t_start_idx, t_end_idx)
    n_t = t_end_idx - t_start_idx
    out = {name: arr[:, sl] for name, arr in cols.items()}
    out["mins_since_last_start"] = since_start
    out["mins_since_last_end"] = since_end
    out["active_for_min"] = np.where(out["active_now"] == 1, since_start, 0.0)

    times = pd.date_range(grid.t0 + t_start_idx * SLOT, periods=n_t, freq=SLOT)
    local = times.tz_convert(KYIV_TZ)
    slot_of_day = (local.hour * 2 + local.minute // 30).to_numpy()
    out["slot_of_day"] = np.tile(slot_of_day, (n_o, 1))
    out["dow"] = np.tile(local.dayofweek.to_numpy(), (n_o, 1))
    out["slot_sin"] = np.sin(2 * np.pi * out["slot_of_day"] / 48)
    out["slot_cos"] = np.cos(2 * np.pi * out["slot_of_day"] / 48)

    t_utc = np.tile(times.tz_convert("UTC").tz_localize(None).to_numpy(), n_o)
    frame = pd.DataFrame({"oblast": np.repeat(grid.oblasts, n_t), "t": pd.DatetimeIndex(t_utc, tz="UTC")})
    for name in feature_columns():
        frame[name] = out[name].ravel(order="C")
    return frame
