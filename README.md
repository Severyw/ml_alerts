# Air raid alert forecasting (Ukraine)

Stage 2 selection mini-project: forecast air-raid alerts per oblast with
time-series methods, evaluated on the most recent period with no data
leakage. Decisions were made step by step with an AI assistant; the full
conversation export is submitted separately.

## What it does

For each of 24 oblast-level units (Luhansk excluded, see below) and every
30-minute slot `t`, it predicts the probability of an alert **1 hour later**.
Two separate models are trained, one per mode, and reported separately:

- **onset** (no alert is active at `t`): will a new alert start in `(t, t+1h]`?
- **continuation** (an alert is active at `t`): is it still active at exactly `t+1h`?

Why separate: the two questions have different base rates, different useful
features and different baselines. A mixed metric would mostly measure how well
we recognise "alert now", which is trivial, and would hide the hard part.
Persistence ("alert now -> alert later") is a constant within a mode, so it
is reported only for completeness.

## Setup and one-command run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh        # downloads the pinned data if missing, then trains and evaluates
```

About 6 minutes on 2 CPUs. Output goes to `reports/`: `results_1h.txt/json`
(every metric with its positive rate), `frozen_config_1h.json` (written
*before* the final fold is fitted), `calibration_1h_final.png`. Runs are
deterministic (fixed seed 42); two runs in the development sandbox gave
byte-identical reports. The committed `reports/` are those outputs.

Tested with Python 3.13, pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1,
matplotlib 3.11.2 (pinned in `requirements.txt`).

## Data and pinned commit

Source: https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset
(MIT-licensed code; the data are scraped from public Telegram channels and
the terms for the data themselves are unclear, so the CSVs are **not**
committed here). `scripts/fetch_data.sh` downloads exactly commit
`22d0443d1d1cdf6a0a413c6d15c3078b5ed26127` (2026-10-15 03:50 UTC), not the
latest state, and verifies the hash. The headline uses the **volunteer** file
(`volunteer_data_en.csv`); the official file is stale (see below) and only
used for cross-checks in `src/data_checks.py`.

## Labels, grid and cutoff

- 30-minute grid in UTC. An event at time `x` becomes visible at slot
  `ceil(x)`. An alert is active at `t` iff `start <= t < end` (half-open).
  Per-raion rows are merged into oblast-level episodes. Kyiv-local slot of
  day and weekday are used only as features.
- **Onset label** (rows with no alert at `t`): 1 if some alert starts in
  `(t, t+H]` (right-closed). An alert that starts exactly at `t` belongs to
  the state at `t` and is not an onset.
- **Continuation label** (rows with an alert at `t`): 1 if the alert state is
  still active at `t+H`, i.e. the alert did not end earlier.
- **Cutoff:** the scraper writes an alert only after it has ended, so the
  newest alerts of a snapshot are missing. The last 12 h of the file are
  unused and the whole label window must lie before that cutoff. For the
  pinned file the last record ends 2026-10-15 03:44 UTC and the last
  forecast time is 2026-10-14 14:30 UTC.
- **Features** use only events with raw timestamp <= `t` (own oblast lags and
  trailing windows, neighbours, national activity, time of day/week). The
  `naive` flag is never a feature. Leakage tests delete, move and extend
  events after `t` and require identical feature rows.

## Evaluation protocol

- Model: scikit-learn `HistGradientBoostingClassifier`, `oblast` as a
  categorical feature, fixed seed, no class weights, no recalibration.
- 3 walk-forward folds of 28 days, expanding training window, a gap of
  `H = 2` slots between the last training slot and the first test slot
  (enforced in `src/splits.py`: a leaky `Fold` cannot be constructed).
  The last ("final") fold is the latest 28 days before the cutoff.
- Hyperparameters and training-window length (365 d / 730 d / full) were
  chosen by Brier score on the two earlier folds only, written to
  `frozen_config_1h.json`, and only then was the final fold fitted.
  Tuning-fold numbers are therefore optimistic.
- Baselines: persistence; per-oblast slot-of-day x day-of-week frequency
  (fitted on the training rows of the fold); recent-24 h frequency (trailing
  features only).

## Results (final fold, 2026-09-16 to 2026-10-14, 1 h horizon)

The positive rate (pos) is shown with every metric. The PR-AUC of a
constant score equals pos.

| Mode | Predictor | n | pos | PR-AUC | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|
| onset | **HGB model** | 28,179 | 25.2% | **0.531** | **0.154** | 0.015 |
| onset | recent 24 h | 28,179 | 25.2% | 0.498 | 0.159 | 0.028 |
| onset | slot x weekday | 28,179 | 25.2% | 0.461 | 0.176 | 0.091 |
| onset | persistence | 28,179 | 25.2% | 0.252 | 0.252 | 0.252 |
| continuation | **HGB model** | 4,077 | 27.9% | **0.561** | **0.170** | 0.018 |
| continuation | slot x weekday | 4,077 | 27.9% | 0.447 | 0.195 | 0.071 |
| continuation | recent 24 h | 4,077 | 27.9% | 0.435 | 0.192 | 0.079 |
| continuation | persistence | 4,077 | 27.9% | 0.279 | 0.721 | 0.721 |

Frozen configs: onset 150 iterations / 31 leaves / last 365 days;
continuation 100 iterations / 15 leaves / last 365 days.

Read these with care:

- **The margin over the recent-24 h baseline in onset mode is small**
  (PR-AUC 0.531 vs 0.498, Brier 0.154 vs 0.159). Most of the model's skill
  in onset mode is already available from "how many onsets happened in the
  last 24 hours". The model's clearer gains are over slot x weekday, and in
  continuation mode.
- **The base rate moves a lot.** The onset positive rate was 17.9%, 22.4% and
  25.2% in the three folds. The model stayed calibrated (ECE 0.015) while the
  slot x weekday baseline, fitted on calmer periods, did not.
- **Per-oblast results are noisy.** `reports/results_1h.txt` has a per-oblast
  table with `n_pos`. Some oblasts have very few positives (e.g. Zakarpatska
  has none in the final fold; Ternopilska continuation has 209 rows), so
  per-oblast PR-AUC should not be over-interpreted.
- **Effective sample size is far below the row count**: rows within a
  28-day block are strongly autocorrelated, and there is only one final fold.
  Tuning differences between configurations are tiny (about 0.0006 Brier in
  onset mode).
- Calibration is reported (`calibration_1h_final.png`, tables in the report),
  never corrected. The 6 h horizon was not run.

## Known data noise

- **Imputed ends.** About 5% of volunteer rows carry `naive = True`, whose end
  time is simply start + 30 min. These are kept (and the flag is never used as
  a feature), so some alert durations are not real.
- **Zero-slot alerts.** An alert shorter than one slot can start and end
  inside a single slot: it is flagged and counted for onset labels but is
  never "active" at any grid time, so it also never appears in the continuation
  mode.
- **`started_at` shift: not measured directly; indirect evidence suggests it is
  masked at oblast level.** (Details below.)
- Official file stale since 2026-09-07 with an outage 2026-08-29..09-01;
  ~38% exact-duplicate rows in its raw CSV (dropped by `load_raw`).
- Luhansk oblast has a permanent siren and is excluded; Crimea and Sevastopol
  are absent from the data.
- Timestamps are treated as UTC: evidence supports this for 2025-26, but 2022-24
  is inconclusive.

## Adjacency

`src/adjacency.py` lists 49 land-border pairs among the 24 units. It was first
drafted by the AI assistant and then **checked pair by pair against a map by
the author**, who confirmed it. Cherkasy-Dnipropetrovsk and
Vinnytsia-Chernivtsi do not border and are not in the list; Kyiv City borders
only Kyiv oblast. Print it with `python -m src.adjacency`.

## Tests

The development sandbox could not install `pytest` (PyPI was blocked). The 49
tests were run there only through a throwaway shim (not in the repo) that
fakes `pytest.raises`, `approx` and `mark.parametrize`; **they have not been
run under real pytest.** To run them for real:

```bash
python -m pytest tests/ -v
```

Key tests: `tests/test_splits.py` (train max t + gap < validation min t),
`tests/test_features.py` (events after `t` cannot change the row at `t`),
`tests/test_labels.py` (ceil boundaries, zero-slot alerts, onset boundary),
`tests/test_train.py` (flipping labels in the final test window changes neither
the frozen config nor the final predictions).

## Details: the `started_at` rule and other data issues (`src/data_checks.py` prints the numbers)

- The official file (`official_data_en.csv`) stopped updating after
  2026-09-07 and is ~5.5 weeks stale as of this writing; the volunteer
  file is current to within a day. The volunteer file is the one used
  for the final evaluation period.
- The official file's raw CSV has ~38% exact-duplicate rows (an
  ingestion artifact); `src/data_io.load_raw` drops them.
- Both source Telegram-channel processors contain a rule that would
  silently shift an alert's `started_at` later if a second "activated"
  message for the same location arrives within 3 hours of the first.
  This is **not measured directly** -- the processed CSV discards the
  original `started_at` with no trace, and we don't have Telegram API
  credentials to re-derive it from the raw messages. `estimate_started_at_shift`
  in `src/data_checks.py` gives an indirect proxy (matching official vs.
  volunteer episodes with the same end time): it suggests the effect is
  masked at oblast-level granularity, most likely because another
  raion's episode is usually already keeping the oblast "active" when
  any one raion's start time gets shifted. This is a plausibility
  argument, not a verified finding, and it would need re-checking before
  relying on raion-level or single-location episode durations as
  features.
- Luhanska oblast (and Crimea/Sevastopol, which never appears in the
  data) has had a permanent, never-logged siren since April 2022 and is
  excluded from the modelling target.
- In December 2025 the official reporting granularity switched from
  oblast-level to raion-level, which changes the positive-rate regime;
  this is treated as a regime break in evaluation, not ignored.
