#!/usr/bin/env python3
"""C2 figure: final hypervolume per run (6 seeds per arm) and in-box rate over real evaluations."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = [("sa_lhfid", "SA-LHFiD", "#eda100"), ("sa_nsga2", "SA-NSGA-II", "#1baf7a"),
        ("random", "Random", "#2a78d6"), ("nsga2", "NSGA-II", "#eb6834")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BOX = {"clip_score": 25.2, "clip_image_fidelity": 75.0, "latency_p50_ms": 600.0, "unet_weight_mb": 1700.0}


def in_box(o: dict) -> bool:
    return (o["clip_score"] >= BOX["clip_score"] and o["clip_image_fidelity"] >= BOX["clip_image_fidelity"]
            and o["latency_p50_ms"] <= BOX["latency_p50_ms"] and o["unet_weight_mb"] <= BOX["unet_weight_mb"])


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c2 = json.loads((ROOT / "artifacts/analysis/c2.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    ax = axes[0]
    for x, (arm, name, color) in enumerate(ARMS):
        values = [r["hypervolume"]["primary_4d"] / 1e6 for r in c2["runs"] if r["arm"] == arm]
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, s=34, color=color, edgecolors="white", linewidths=1.2, zorder=3)
        ax.hlines(statistics.median(values), x - 0.25, x + 0.25, color=INK, linewidth=2, zorder=4)
    ax.set_xticks(range(len(ARMS)), [name for _, name, _ in ARMS], fontsize=9)
    ax.set_ylabel("Hypervolume (×10⁶), 150 real evaluations", color=MUTED, fontsize=9)
    ax.set_title("Final hypervolume per seed (bar = median)", color=INK, fontsize=10, loc="left")
    ax = axes[1]
    for arm, name, color in ARMS:
        curves = []
        for seed in range(1, 7):
            for root in (ROOT / "artifacts/nas_c2", ROOT / "artifacts/nas"):
                path = root / f"seed{seed}" / arm / "search_results.json"
                if path.exists() and not (arm in ("random", "nsga2") and root.name == "nas" and seed > 3):
                    candidates = json.loads(path.read_text())["candidates"]
                    curves.append(np.cumsum([in_box(c["objectives"]) for c in candidates]))
                    break
        mean = np.mean(curves, axis=0)
        ax.plot(np.arange(1, len(mean) + 1), mean, color=color, linewidth=2, label=name)
    ax.axvline(50, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate("surrogate rounds start", (50, 0), xytext=(4, 4), textcoords="offset points", color=MUTED, fontsize=8)
    ax.set_xlim(0, 155)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    ax.set_xlabel("Real evaluations", color=MUTED, fontsize=9)
    ax.set_ylabel("Candidates inside the reference box (mean of 6 seeds)", color=MUTED, fontsize=9)
    ax.set_title("Where the real budget goes", color=INK, fontsize=10, loc="left")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=8)
        a.grid(color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
    fig.savefig(ROOT / "docs/figures/c2_sa_lhfid.png", dpi=200)
    print("docs/figures/c2_sa_lhfid.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
