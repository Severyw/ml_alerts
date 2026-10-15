"""Metrics. Every summary carries n, n_pos and the positive rate next to the
numbers, because PR-AUC and Brier are only interpretable relative to the base
rate (and the base rate moves a lot between periods)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


def pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision. NaN if there are no positives. For a constant score
    (e.g. persistence within one mode) this equals the positive rate."""
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table with (up to) n_bins equal-count bins of the predicted
    probability: mean predicted vs observed frequency. Report-only; nothing is
    fitted on it."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:  # constant predictions -> one bin
        ids = np.zeros(len(p), dtype=int)
    else:
        ids = np.searchsorted(edges[1:-1], p, side="right")
    df = pd.DataFrame({"bin": ids, "y": y, "p": p})
    out = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean")).reset_index(drop=True)
    return out


def expected_calibration_error(table: pd.DataFrame) -> float:
    n = table["n"].sum()
    return float((table["n"] * (table["obs_rate"] - table["mean_pred"]).abs()).sum() / n)


def summarize(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y)
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "pos_rate": float(y.mean()) if len(y) else float("nan"),
        "pr_auc": pr_auc(y, p),
        "brier": brier(y, p),
        "ece": expected_calibration_error(calibration_table(y, p)),
    }


def per_oblast(oblast: np.ndarray, y: np.ndarray, preds: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per oblast: n, n_pos, pos_rate and, per named prediction vector,
    PR-AUC and Brier. Oblasts with few positives give noisy PR-AUC; read n_pos."""
    rows = []
    oblast = np.asarray(oblast)
    y = np.asarray(y)
    for o in sorted(set(oblast)):
        m = oblast == o
        row = {"oblast": o, "n": int(m.sum()), "n_pos": int(y[m].sum()), "pos_rate": float(y[m].mean())}
        for name, p in preds.items():
            row[f"{name}_pr_auc"] = pr_auc(y[m], np.asarray(p)[m])
            row[f"{name}_brier"] = brier(y[m], np.asarray(p)[m])
        rows.append(row)
    return pd.DataFrame(rows)
