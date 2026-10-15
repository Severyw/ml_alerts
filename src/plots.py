"""Calibration (reliability) figure. Colours are slots 1-3 of the validated
categorical palette in light mode (blue / orange / aqua); the aqua slot is
below 3:1 contrast on the light surface, so every series is also direct-labelled
and the same numbers are available as tables in the text report."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = {  # name -> (colour, marker)
    "model": ("#2a78d6", "o"),
    "slot x weekday": ("#eb6834", "s"),
    "recent 24 h": ("#1baf7a", "^"),
}


def calibration_figure(
    tables: dict[str, dict[str, pd.DataFrame]],
    pos_rates: dict[str, float],
    title: str,
    path: pathlib.Path,
) -> None:
    """tables[mode][series_name] = calibration_table(...) output."""
    modes = list(tables)
    fig, axes = plt.subplots(1, len(modes), figsize=(5.6 * len(modes), 5.4), facecolor=SURFACE)
    axes = [axes] if len(modes) == 1 else list(axes)
    for ax, mode in zip(axes, modes):
        ax.set_facecolor(SURFACE)
        hi = max(max(t["mean_pred"].max(), t["obs_rate"].max()) for t in tables[mode].values())
        hi = max(0.05, min(1.0, hi * 1.08))
        ax.plot([0, hi], [0, hi], color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(hi * 0.98, hi * 0.03, "dashed line = perfect calibration", color=INK_2, fontsize=8, ha="right")
        for name, t in tables[mode].items():
            colour, marker = SERIES[name]
            ax.plot(t["mean_pred"], t["obs_rate"], color=colour, lw=1.5, marker=marker, ms=6,
                    mec=SURFACE, mew=1.2, label=name, zorder=3)
            last = t.iloc[-1]
            ax.annotate(name, (last["mean_pred"], last["obs_rate"]), xytext=(6, -3 if name != "model" else 6),
                        textcoords="offset points", fontsize=8, color=INK_2)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("mean predicted probability (equal-count bins)", color=INK_2, fontsize=9)
        ax.set_ylabel("observed frequency", color=INK_2, fontsize=9)
        ax.set_title(f"{mode} mode  (positive rate {pos_rates[mode]:.1%})", color=INK, fontsize=11, loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        ax.tick_params(colors=INK_2, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
