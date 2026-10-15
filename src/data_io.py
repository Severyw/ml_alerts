"""Loading and basic cleaning of the air-raid-sirens dataset.

Two files are used:
  - official_data_en.csv   : alerts reported on the official "air_alert_ua"
                             Telegram channel, at oblast/raion/hromada level.
  - volunteer_data_en.csv  : alerts reported by volunteers (eTryvoga /
                             "UkraineAlarmSignal" channel), at oblast level
                             only, going back further.

Both files can contain more than one row per oblast at the same moment
(e.g. several raions of the same oblast in alert at once). For an
oblast-level target we collapse each oblast's rows into non-overlapping
"any alert in this oblast" episodes with `union_intervals`.

LUHANSK AND CRIMEA/SEVASTOPOL
Luhanska oblast and Crimea/Sevastopol have had a *permanent* siren since
2022 (see datasets/README.md in the source repo) that is never logged as
a normal start/end event. Luhanska rows exist only for the first few weeks
of the war; Crimea/Sevastopol do not appear at all. Per the agreed spec we
drop Luhansk from the modelling target entirely rather than pretend it has
a meaningful on/off state.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import EXCLUDED_OBLASTS


@dataclass(frozen=True)
class RawData:
    official: pd.DataFrame
    volunteer: pd.DataFrame
    official_path: pathlib.Path
    volunteer_path: pathlib.Path


def load_raw(data_dir: str | pathlib.Path) -> RawData:
    """Read both CSVs, parse timestamps as UTC, drop exact-duplicate rows,
    and standardise the oblast-name column to `oblast`."""
    data_dir = pathlib.Path(data_dir)
    off_path = data_dir / "official_data_en.csv"
    vol_path = data_dir / "volunteer_data_en.csv"

    official = pd.read_csv(off_path)
    volunteer = pd.read_csv(vol_path)

    for df in (official, volunteer):
        for col in ("started_at", "finished_at"):
            df[col] = pd.to_datetime(df[col], utc=True)

    volunteer = volunteer.rename(columns={"region": "oblast"})

    n_before = len(official)
    official = official.drop_duplicates().reset_index(drop=True)
    n_after = len(official)
    if n_before != n_after:
        # Known issue: the repo's CSV has been re-appended with overlapping
        # history at least once, producing exact full-row duplicates.
        # We silently dedupe here; data_checks.py reports the count so it
        # is never a silent surprise.
        pass

    return RawData(official, volunteer, off_path, vol_path)


def union_intervals(df: pd.DataFrame, key: str = "oblast") -> pd.DataFrame:
    """Collapse possibly-overlapping (started_at, finished_at) rows within
    each `key` group into the minimal set of non-overlapping "something is
    active" episodes. This is how several raion-level rows in the official
    file become one oblast-level "in alert" episode.
    """
    out = []
    for group_key, g in df.sort_values("started_at").groupby(key):
        starts = g.started_at.to_numpy()
        ends = g.finished_at.to_numpy()
        cur_s, cur_e = starts[0], ends[0]
        for s, e in zip(starts[1:], ends[1:]):
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                out.append((group_key, cur_s, cur_e))
                cur_s, cur_e = s, e
        out.append((group_key, cur_s, cur_e))
    result = pd.DataFrame(out, columns=[key, "started_at", "finished_at"])
    result["started_at"] = pd.to_datetime(result["started_at"], utc=True)
    result["finished_at"] = pd.to_datetime(result["finished_at"], utc=True)
    return result


def drop_excluded_oblasts(df: pd.DataFrame, key: str = "oblast") -> pd.DataFrame:
    return df[~df[key].isin(EXCLUDED_OBLASTS)].reset_index(drop=True)
