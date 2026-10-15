"""Walk-forward training and evaluation of the onset / continuation models.

    python -m src.train --data-dir data/raw --horizon 1h --out reports

Design (agreed in the chat log):
* Data: the volunteer file (the official file stops on 2026-09-07). Luhansk excluded.
* Two SEPARATE models per horizon, one per mode: "onset" (rows with no alert
  at t) and "continuation" (rows with an alert at t). Their metrics are never mixed.
* Model: scikit-learn HistGradientBoostingClassifier, `oblast` as a categorical
  feature. No class weights (we want probabilities), no early stopping (it would
  use a random validation split of autocorrelated rows), fixed seed.
* Folds: walk-forward, expanding window, at most 3 folds, gap >= H between the
  last training slot and the first test slot (see src/splits.py). The LAST fold
  ("final") ends at the 12 h cutoff of the snapshot.
* Tuning uses ONLY the folds before the final one. `run_experiment` receives
  `folds[:-1]` for tuning by construction, then calls `on_frozen(...)`, which
  `main` uses to write reports/frozen_config_<h>.json BEFORE the final fold
  is fitted or scored. Selection criterion: mean Brier over the tuning folds
  (a proper scoring rule).
* The training-window length (last 365 d / 730 d / full history) is one of the
  tuned dimensions, so "full history vs last 1-2 years" is decided on the
  tuning folds. A post-hoc comparison on the final fold is also reported, for
  information only, never for selection.
* Calibration: reported (reliability tables, ECE, plot). Nothing is recalibrated.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from src import baselines as bl
from src.adjacency import adjacency
from src.config import GRID_START, HORIZONS, MODELLED_OBLASTS, PINNED_COMMIT, SEED, SLOT
from src.data_io import drop_excluded_oblasts, load_raw, union_intervals
from src.evaluate import brier, calibration_table, per_oblast, summarize
from src.features import WINDOWS, build_feature_frame, feature_columns
from src.labels import build_grid, build_label_frame
from src.plots import calibration_figure
from src.splits import Fold, last_usable_t_idx, make_walk_forward_folds, mask_in_range

MODES = ("onset", "continuation")
SLOTS_PER_DAY = 48


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelConfig:
    max_iter: int
    max_leaf_nodes: int
    window_days: int | None          # None = all history available to the fold
    learning_rate: float = 0.1
    min_samples_leaf: int = 200      # large leaves: rows are strongly autocorrelated
    l2_regularization: float = 1.0

    @property
    def window_slots(self) -> int | None:
        return None if self.window_days is None else self.window_days * SLOTS_PER_DAY


DEFAULT_GRID = [
    ModelConfig(it, leaf, window)
    for (it, leaf) in ((100, 15), (150, 31), (200, 63))
    for window in (365, 730, None)
]


@dataclass
class Dataset:
    X: pd.DataFrame            # float32 features + categorical `oblast`
    y: np.ndarray              # int8 target for the chosen horizon (label matching each row's mode)
    mode: np.ndarray           # "onset" | "continuation"
    t_idx: np.ndarray          # slot index of the forecast time
    oblast: np.ndarray         # str
    horizon_name: str
    horizon_slots: int
    t0: pd.Timestamp
    first_t_idx: int
    last_t_idx: int


def prepare_dataset(df: pd.DataFrame, horizon_name: str, t0: pd.Timestamp = GRID_START) -> Dataset:
    """df: alert rows already filtered to the modelled oblasts."""
    h = HORIZONS[horizon_name]
    last_end = df["finished_at"].max()
    last_t = last_usable_t_idx(last_end, t0, h)
    n_steps = int((last_end.floor(SLOT) - t0) // SLOT)
    first_t = max(WINDOWS)  # full 7-day windows for every row
    grid = build_grid(df, t0, n_steps, oblasts=MODELLED_OBLASTS)
    feats = build_feature_frame(grid, union_intervals(df), adjacency(), first_t, last_t + 1)
    labels = build_label_frame(grid, first_t, last_t + 1, {horizon_name: h})
    X = feats[feature_columns()].astype("float32")
    X["oblast"] = pd.Categorical(feats["oblast"], categories=list(MODELLED_OBLASTS))
    return Dataset(
        X=X,
        y=labels[f"target_{horizon_name}"].to_numpy(dtype=np.int8),
        mode=labels["mode"].to_numpy(),
        t_idx=((feats["t"] - t0) // SLOT).to_numpy(dtype=np.int64),
        oblast=feats["oblast"].to_numpy(),
        horizon_name=horizon_name,
        horizon_slots=h,
        t0=t0,
        first_t_idx=first_t,
        last_t_idx=last_t,
    )


def fit_predict(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, cfg: ModelConfig) -> np.ndarray:
    if len(np.unique(y_tr)) < 2:  # degenerate fold: constant prediction
        return np.full(len(X_te), float(np.mean(y_tr)) if len(y_tr) else 0.0)
    clf = HistGradientBoostingClassifier(
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        categorical_features="from_dtype",
        early_stopping=False,
        random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]


def _rows(ds: Dataset, mode: str, first: int, last: int) -> np.ndarray:
    return (ds.mode == mode) & mask_in_range(ds.t_idx, first, last)


def _baseline_preds(ds: Dataset, mode: str, tr: np.ndarray, te: np.ndarray) -> dict[str, np.ndarray]:
    slot = ds.X["slot_of_day"].to_numpy()
    dow = ds.X["dow"].to_numpy()
    sd = bl.SlotDowFrequency().fit(ds.oblast[tr], slot[tr], dow[tr], ds.y[tr])
    return {
        "persistence": bl.persistence(mode, int(te.sum())),
        "slot_dow": sd.predict(ds.oblast[te], slot[te], dow[te]),
        "recent_24h": bl.recent_frequency_24h(
            mode, ds.X["onset_cnt_48"].to_numpy()[te], ds.X["active_frac_48"].to_numpy()[te], ds.horizon_slots
        ),
    }


def _ts(ds: Dataset, idx: int) -> str:
    return str(ds.t0 + idx * SLOT)


# --------------------------------------------------------------------------- #
def run_experiment(
    ds: Dataset,
    folds: list[Fold],
    cfgs: list[ModelConfig],
    on_frozen: Callable[[str, ModelConfig, list[dict]], None] | None = None,
    compare_windows: bool = True,
    log: Callable[[str], None] = print,
) -> tuple[dict, dict]:
    """Returns (results, arrays). `results` is JSON-able; `arrays` holds the
    final-fold row-level vectors for the per-oblast table and the plot."""
    tune_folds, final = folds[:-1], folds[-1]
    assert final.name == "final" and all(f.test_last_t < final.test_first_t for f in tune_folds)
    results: dict = {"modes": {}}
    arrays: dict = {}

    for mode in MODES:
        cache: dict[tuple[ModelConfig, str], np.ndarray] = {}

        # ---- 1. tuning: tuning folds only ------------------------------------
        tuning_rows = []
        for cfg in cfgs:
            briers = []
            for f in tune_folds:
                tr = _rows(ds, mode, *f.train_range(cfg.window_slots))
                te = _rows(ds, mode, f.test_first_t, f.test_last_t)
                p = fit_predict(ds.X[tr], ds.y[tr], ds.X[te], cfg)
                cache[(cfg, f.name)] = p
                briers.append(brier(ds.y[te], p))
            tuning_rows.append({**asdict(cfg), "brier_per_tuning_fold": briers, "brier_mean": float(np.mean(briers))})
            log(f"  [{mode}] {asdict(cfg)} mean Brier on tuning folds = {np.mean(briers):.5f}")
        best_i = int(np.argmin([r["brier_mean"] for r in tuning_rows]))
        frozen = cfgs[best_i]

        # ---- 2. freeze BEFORE the final fold is touched ----------------------
        if on_frozen is not None:
            on_frozen(mode, frozen, tuning_rows)
        log(f"  [{mode}] FROZEN: {asdict(frozen)}")

        # ---- 3. score every fold with the frozen config ---------------------
        fold_results = []
        for f in folds:
            tr = _rows(ds, mode, *f.train_range(frozen.window_slots))
            te = _rows(ds, mode, f.test_first_t, f.test_last_t)
            is_tuning = f.name != "final"
            p = cache[(frozen, f.name)] if is_tuning else fit_predict(ds.X[tr], ds.y[tr], ds.X[te], frozen)
            preds = {"model": p, **_baseline_preds(ds, mode, tr, te)}
            first_tr, last_tr = f.train_range(frozen.window_slots)
            entry = {
                "fold": f.name,
                "used_for_tuning": is_tuning,
                "train_t_utc": [_ts(ds, first_tr), _ts(ds, last_tr)],
                "test_t_utc": [_ts(ds, f.test_first_t), _ts(ds, f.test_last_t)],
                "gap_slots": f.gap_slots,
                "n_train": int(tr.sum()),
                "train_pos_rate": float(ds.y[tr].mean()),
                "metrics": {name: summarize(ds.y[te], q) for name, q in preds.items()},
                "calibration_model": calibration_table(ds.y[te], p).round(5).to_dict("records"),
            }
            fold_results.append(entry)
            if not is_tuning:
                arrays[mode] = {
                    "oblast": ds.oblast[te], "y": ds.y[te], "preds": preds,
                    "tables": {n: calibration_table(ds.y[te], q) for n, q in preds.items() if n != "persistence"},
                }
                entry["calibration_all"] = {n: t.round(5).to_dict("records") for n, t in arrays[mode]["tables"].items()}
                entry["per_oblast"] = per_oblast(
                    ds.oblast[te], ds.y[te], {"model": preds["model"], "slot_dow": preds["slot_dow"]}
                ).round(5).to_dict("records")

        # ---- 4. post-hoc, information only: training-window length on the final fold
        window_cmp = []
        if compare_windows:
            te = _rows(ds, mode, final.test_first_t, final.test_last_t)
            for w in sorted({c.window_days for c in cfgs}, key=lambda v: (v is None, v or 0)):
                cfg_w = ModelConfig(frozen.max_iter, frozen.max_leaf_nodes, w)
                tr = _rows(ds, mode, *final.train_range(cfg_w.window_slots))
                p = (arrays[mode]["preds"]["model"] if w == frozen.window_days
                     else fit_predict(ds.X[tr], ds.y[tr], ds.X[te], cfg_w))
                s = summarize(ds.y[te], p)
                window_cmp.append({"window_days": w, "n_train": int(tr.sum()), **{k: s[k] for k in ("pos_rate", "pr_auc", "brier", "ece")}})
            log(f"  [{mode}] window comparison (final fold, information only): {window_cmp}")

        results["modes"][mode] = {
            "frozen_config": asdict(frozen),
            "tuning_table": tuning_rows,
            "folds": fold_results,
            "window_comparison_final_fold_post_hoc": window_cmp,
        }
    return results, arrays


# --------------------------------------------------------------------------- #
def format_report(results: dict, meta: dict) -> str:
    L: list[str] = []
    a = L.append
    a(f"RESULTS  horizon={meta['horizon']}  (volunteer data, pinned commit {meta['pinned_commit'][:12]})")
    a(f"last record in file : {meta['last_record_end']}")
    a(f"12 h cutoff         : last usable forecast time t = {meta['last_t_utc']} (label window (t, t+H] ends before the cutoff)")
    a(f"folds               : {meta['n_folds']} x {meta['test_days']} days, gap = {meta['gap_slots']} slots (>= H), expanding training")
    a("Positive rate (pos) is shown beside every metric. PR-AUC of a constant score equals pos.")
    a("")
    for mode, r in results["modes"].items():
        a("=" * 100)
        a(f"MODE: {mode}   frozen config (chosen on tuning folds only): {r['frozen_config']}")
        a("=" * 100)
        for fo in r["folds"]:
            tag = "TUNING fold (optimistic: used to pick config)" if fo["used_for_tuning"] else "FINAL fold (headline)"
            a(f"-- {fo['fold']}: {tag}")
            a(f"   train {fo['train_t_utc'][0]} .. {fo['train_t_utc'][1]}  n_train={fo['n_train']:,}  train_pos={fo['train_pos_rate']:.4f}")
            a(f"   test  {fo['test_t_utc'][0]} .. {fo['test_t_utc'][1]}")
            a(f"   {'predictor':<12}{'n':>8}{'n_pos':>8}{'pos':>9}{'PR-AUC':>9}{'Brier':>9}{'ECE':>9}")
            for name, m in fo["metrics"].items():
                a(f"   {name:<12}{m['n']:>8,}{m['n_pos']:>8,}{m['pos_rate']:>9.4f}{m['pr_auc']:>9.4f}{m['brier']:>9.4f}{m['ece']:>9.4f}")
            a("")
        fin = r["folds"][-1]
        a(f"Calibration tables, final fold (equal-count bins; report only, nothing recalibrated):")
        for name, rows in fin["calibration_all"].items():
            a(f"   {name}:")
            a("      " + "  ".join(f"[{x['mean_pred']:.3f}->{x['obs_rate']:.3f} n={x['n']}]" for x in rows))
        a("")
        a("Per-oblast, final fold (read n_pos: PR-AUC is noisy when it is small):")
        a(f"   {'oblast':<26}{'n':>7}{'n_pos':>7}{'pos':>8}{'mdl PR':>8}{'mdl Br':>8}{'sd PR':>8}{'sd Br':>8}")
        for x in fin["per_oblast"]:
            a(f"   {x['oblast']:<26}{x['n']:>7,}{x['n_pos']:>7,}{x['pos_rate']:>8.3f}{x['model_pr_auc']:>8.3f}{x['model_brier']:>8.4f}{x['slot_dow_pr_auc']:>8.3f}{x['slot_dow_brier']:>8.4f}")
        a("")
        a("Training-window length on the final fold (POST-HOC, information only; selection used the tuning folds):")
        for w in r["window_comparison_final_fold_post_hoc"]:
            a(f"   window={str(w['window_days']):>5} d  n_train={w['n_train']:>9,}  pos={w['pos_rate']:.4f}  PR-AUC={w['pr_auc']:.4f}  Brier={w['brier']:.4f}  ECE={w['ece']:.4f}")
        a("")
        a("Tuning table (mean Brier over tuning folds; lower is better):")
        for t in sorted(r["tuning_table"], key=lambda z: z["brier_mean"]):
            a(f"   iter={t['max_iter']:>3} leaves={t['max_leaf_nodes']:>3} window={str(t['window_days']):>5}  Brier={t['brier_mean']:.5f}  per-fold={[round(b, 5) for b in t['brier_per_tuning_fold']]}")
        a("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--horizon", default="1h", choices=list(HORIZONS))
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--test-days", type=int, default=28)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    raw = load_raw(args.data_dir)
    df = drop_excluded_oblasts(raw.volunteer)
    ds = prepare_dataset(df, args.horizon)
    gap = ds.horizon_slots
    folds = make_walk_forward_folds(ds.last_t_idx, ds.first_t_idx, args.n_folds, args.test_days * SLOTS_PER_DAY, gap)
    print(f"dataset: {len(ds.y):,} rows; folds: " + "; ".join(
        f"{f.name} test {_ts(ds, f.test_first_t)}..{_ts(ds, f.test_last_t)}" for f in folds))

    frozen_path = out / f"frozen_config_{args.horizon}.json"
    frozen: dict = {}

    def on_frozen(mode: str, cfg: ModelConfig, tuning: list[dict]) -> None:
        frozen[mode] = {"config": asdict(cfg), "tuning_table": tuning}
        frozen["written_utc"] = str(pd.Timestamp.now(tz="UTC"))
        frozen["note"] = "Written before the final fold was fitted or scored."
        frozen_path.write_text(json.dumps(frozen, indent=2))

    results, arrays = run_experiment(ds, folds, DEFAULT_GRID, on_frozen=on_frozen)

    meta = {
        "horizon": args.horizon, "pinned_commit": PINNED_COMMIT, "seed": SEED,
        "last_record_end": str(df["finished_at"].max()), "last_t_utc": _ts(ds, ds.last_t_idx),
        "n_folds": args.n_folds, "test_days": args.test_days, "gap_slots": gap,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "runtime_s": round(time.time() - t_start, 1),
    }
    results["meta"] = meta
    (out / f"results_{args.horizon}.json").write_text(json.dumps(results, indent=2, default=float))
    report = format_report(results, meta)
    (out / f"results_{args.horizon}.txt").write_text(report)
    print(report)

    fin = {m: results["modes"][m]["folds"][-1] for m in MODES}
    calibration_figure(
        {m: {"model": arrays[m]["tables"]["model"], "slot x weekday": arrays[m]["tables"]["slot_dow"],
             "recent 24 h": arrays[m]["tables"]["recent_24h"]} for m in MODES},
        {m: fin[m]["metrics"]["model"]["pos_rate"] for m in MODES},
        f"Calibration on the final fold ({args.horizon} horizon), report only",
        out / f"calibration_{args.horizon}_final.png",
    )
    print(f"\nwrote {frozen_path.name}, results_{args.horizon}.json/.txt, calibration_{args.horizon}_final.png to {out}/  ({meta['runtime_s']} s)")


if __name__ == "__main__":
    main()
