"""Single source of truth for constants shared across the project.

Conventions agreed in the project spec (see chat log / README):
* All arithmetic is in UTC. Kyiv-local time is used only to derive
  calendar features (slot-of-day, day-of-week), never for grid arithmetic.
* The grid has 30-minute slots. A raw timestamp x is only visible from
  slot ceil(x) onward (16:20 -> 16:30, 16:30 -> 16:30, 16:31 -> 17:00).
  See src/labels.py:build_grid for why this is leakage-safe.
* Onset label for horizon H: no alert at t -> a new alert starts in (t, t+H].
* Continuation label for horizon H: alert active at t -> still active at
  exactly t+H. (NOT "active at any point in (t, t+H]" -- that definition
  is nearly always true and was rejected; see chat log.)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- paths -------------------------------------------------------------- #
DATA_DIR = REPO_ROOT / "data" / "raw"
OFFICIAL_CSV = "official_data_en.csv"
VOLUNTEER_CSV = "volunteer_data_en.csv"

# --- data source (pinned for reproducibility) ---------------------------- #
DATA_REPO_URL = "https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset.git"
# Full 40-character git SHA, NOT an abbreviation and NOT a branch name --
# scripts/fetch_data.sh checks out exactly this commit, not "latest".
# Commit time: 2026-10-15T03:50:46+00:00 ("Add new data").
PINNED_COMMIT = "22d0443d1d1cdf6a0a413c6d15c3078b5ed26127"
assert len(PINNED_COMMIT) == 40, "PINNED_COMMIT must be a full git SHA (40 hex chars)"

# --- reproducibility ------------------------------------------------------ #
SEED = 42

# --- grid and labels ------------------------------------------------------ #
SLOT = pd.Timedelta("30min")
HORIZONS = {"1h": 2, "6h": 12}  # horizon name -> number of 30-minute slots
GRID_START = pd.Timestamp("2022-03-15", tz="UTC")  # start of the modelled grid; the volunteer file starts 2022-02-25, those first 18 days are not used

# --- data-quality decisions (see README "Data caveats" for the reasoning) - #
# Luhanska oblast (and Crimea/Sevastopol, absent from both files entirely)
# has had a permanent siren since 2022-04 that is never logged as a normal
# start/end event, so its series would be "always off" in the data but
# "always on" in reality. Excluded from the modelling target everywhere.
EXCLUDED_OBLASTS = ("Luhanska oblast",)

# The fixed universe of oblast-level units we model (the 25 names that occur
# in both CSVs, minus EXCLUDED_OBLASTS). Fixed here -- not derived from the
# data -- so array shapes and feature columns cannot change because some
# oblast happens to have no events in a given slice of data.
MODELLED_OBLASTS = (
    "Cherkaska oblast",
    "Chernihivska oblast",
    "Chernivetska oblast",
    "Dnipropetrovska oblast",
    "Donetska oblast",
    "Ivano-Frankivska oblast",
    "Kharkivska oblast",
    "Khersonska oblast",
    "Khmelnytska oblast",
    "Kirovohradska oblast",
    "Kyiv City",
    "Kyivska oblast",
    "Lvivska oblast",
    "Mykolaivska oblast",
    "Odeska oblast",
    "Poltavska oblast",
    "Rivnenska oblast",
    "Sumska oblast",
    "Ternopilska oblast",
    "Vinnytska oblast",
    "Volynska oblast",
    "Zakarpatska oblast",
    "Zaporizka oblast",
    "Zhytomyrska oblast",
)

# The ingestion scraper only writes an alert once it has ENDED, so any alert
# still open at download time is entirely missing from the file -- not an
# absence of alerts, an absence of data. We treat the last CUTOFF_MARGIN
# before a snapshot's last timestamp as unusable for evaluation.
CUTOFF_MARGIN = pd.Timedelta("12h")
