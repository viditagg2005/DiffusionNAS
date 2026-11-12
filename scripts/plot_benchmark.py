#!/usr/bin/env python3
"""Figure for the many-objective optimizer benchmark: archive HV vs evaluations and final-population HV."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"random": "#2a78d6", "nsga2": "#eb6834", "nsga3": "#1baf7a", "lhfid": "#eda100", "lhfid_equal": "#e87ba4"}
NAMES = {"random": "Random", "nsga2": "NSGA-II", "nsga3": "NSGA-III", "lhfid": "LHFiD (p = 1/w)", "lhfid_equal": "LHFiD (p = 1/M)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads((ROOT / "artifacts/surrogate_benchmark/results.json").read_text())
    checkpoints = results["protocol"]["checkpoints"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for ax, key in zip(axes[:2], ("M4", "M6")):
        for algorithm in results["protocol"]["algorithms"]:
            runs = [r for r in results["runs"] if r["objective_set"] == key and r["algorithm"] == algorithm]
            medians = [statistics.median(r["archive_hv"][str(c)] for r in runs) for c in checkpoints]
            ax.plot(checkpoints, medians, color=COLORS[algorithm], linewidth=2, marker="o", markersize=4, label=NAMES[algorithm])
        ax.set_xscale("log")
        ax.set_xticks(checkpoints, [str(c) for c in checkpoints])
        ax.axvline(150, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate("real budget", (150, ax.get_ylim()[0]), xytext=(4, 6), textcoords="offset points", color=MUTED, fontsize=8)
        ax.set_title(f"{key[1]} objectives: archive hypervolume (median of 35)", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Evaluations (log scale)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[2]
    algorithms = ["nsga2", "nsga3", "lhfid", "lhfid_equal"]
    for offset, key in ((-0.2, "M4"), (0.2, "M6")):
        for i, algorithm in enumerate(algorithms):
            values = [r["final_population_hv"] for r in results["runs"] if r["objective_set"] == key and r["algorithm"] == algorithm]
            ax.bar(i + offset, statistics.median(values), width=0.36, color=COLORS[algorithm],
                   alpha=1.0 if key == "M6" else 0.45, edgecolor="white", linewidth=1)
    ax.set_xticks(range(len(algorithms)), [NAMES[a].replace(" (", "\n(") for a in algorithms], fontsize=8)
    ax.set_title("Final-population hypervolume\n(light: 4 objectives, solid: 6 objectives)", color=INK, fontsize=10, loc="left")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
        a.grid(color=GRID, linewidth=0.8, axis="y")
        a.set_axisbelow(True)
    fig.savefig(ROOT / "docs/figures/optimizer_benchmark.png", dpi=200)
    print("docs/figures/optimizer_benchmark.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
