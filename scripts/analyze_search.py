#!/usr/bin/env python3
"""Analyse the real v3 searches: hypervolume per seed and optimizer, drift-free variant, fronts.

Writes artifacts/analysis/search_v3.json and figures to docs/figures/.
- HV (pre-registered 4-D reference: CLIP 25.2, fidelity 75, 600 ms, 1,700 MB) recomputed exactly.
- Drift-free HV: latency replaced by GFLOPs/image (deterministic; reference 36,000 GFLOPs).
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas.surrogate import load_archive  # noqa: E402

FIG = ROOT / "docs" / "figures"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LATENCY_SET = [("clip_score", "max", 25.2), ("clip_image_fidelity", "max", 75.0),
               ("latency_p50_ms", "min", 600.0), ("unet_weight_mb", "min", 1700.0)]
GFLOP_SET = [("clip_score", "max", 25.2), ("clip_image_fidelity", "max", 75.0),
             ("gflops_per_image", "min", 36000.0), ("unet_weight_mb", "min", 1700.0)]


def hypervolume(rows, spec) -> float:
    import moocore

    f = np.array([[-r["objectives"][n] if d == "max" else r["objectives"][n] for n, d, _ in spec] for r in rows])
    ref = np.array([-v if d == "max" else v for _, d, v in spec])
    inside = f[(f < ref).all(axis=1)]
    return float(moocore.hypervolume(inside, ref=ref)) if len(inside) else 0.0


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_archive(ROOT / "artifacts" / "nas")
    checkpoints = [10, 25, 50, 75, 100, 125, 150]
    table, curves = {}, {}
    for seed in ("seed1", "seed2", "seed3"):
        for optimizer in ("random", "nsga2"):
            ordered = [r for r in rows if r["group"] == seed and r["optimizer"] == optimizer]
            for label, spec in (("latency", LATENCY_SET), ("gflops", GFLOP_SET)):
                curve = [hypervolume(ordered[:c], spec) for c in checkpoints]
                curves[(seed, optimizer, label)] = curve
                table.setdefault(label, {}).setdefault(seed, {})[optimizer] = curve[-1]
    summary = {}
    for label in ("latency", "gflops"):
        diffs = [table[label][s]["nsga2"] - table[label][s]["random"] for s in ("seed1", "seed2", "seed3")]
        summary[label] = {
            "per_seed": table[label],
            "nsga2_minus_random": diffs,
            "nsga2_wins": sum(d > 0 for d in diffs),
            "mean": {o: statistics.fmean(table[label][s][o] for s in table[label]) for o in ("random", "nsga2")},
            "sd": {o: statistics.pstdev(table[label][s][o] for s in table[label]) for o in ("random", "nsga2")},
        }
    out = ROOT / "artifacts" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "search_v3.json").write_text(json.dumps(summary, indent=2) + "\n")
    for label, s in summary.items():
        print(label, {k: [f"{x:.4g}" for x in v] if isinstance(v, list) else v for k, v in s.items() if k != "per_seed"})

    # Figure: hypervolume vs evaluations, both indicator variants.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, (label, title) in zip(axes, (("latency", "HV with p50 latency"), ("gflops", "Drift-free HV with GFLOPs"))):
        for optimizer, color, name in (("random", BLUE, "Random"), ("nsga2", ORANGE, "NSGA-II")):
            series = np.array([curves[(s, optimizer, label)] for s in ("seed1", "seed2", "seed3")])
            ax.fill_between(checkpoints, series.min(axis=0), series.max(axis=0), color=color, alpha=0.15, linewidth=0)
            ax.plot(checkpoints, series.mean(axis=0), color=color, linewidth=2, label=name)
            ax.annotate(name, (checkpoints[-1], series.mean(axis=0)[-1]), xytext=(4, 0), textcoords="offset points",
                        color=INK, fontsize=9, va="center")
        style(ax)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Unique candidates evaluated", color=MUTED, fontsize=9)
        ax.set_xlim(0, 175)
    axes[0].set_ylabel("Hypervolume (mean, band = 3-seed range)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.savefig(FIG / "search_hypervolume.png", dpi=200)

    # Figure: pooled candidates, two 2-D projections; eligible and locked finalists highlighted.
    finalists = json.loads((ROOT / "configs" / "generated" / "validation_v3" / "locked_finalists.json").read_text())["finalists"]
    eligible = lambda r: r["objectives"]["clip_score"] >= 25.2 and r["objectives"]["clip_image_fidelity"] >= 75
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, (x, xlabel) in zip(axes, (("latency_p50_ms", "p50 latency, ms (normalized to job reference)"),
                                      ("unet_weight_mb", "U-Net weights, MB"))):
        other = [r for r in rows if not eligible(r)]
        good = [r for r in rows if eligible(r)]
        ax.scatter([r["objectives"][x] for r in other], [r["objectives"]["clip_score"] for r in other], s=10,
                   color="#b9b8b2", linewidths=0, label="Below quality floor")
        ax.scatter([r["objectives"][x] for r in good], [r["objectives"]["clip_score"] for r in good], s=14,
                   color=BLUE, linewidths=0, alpha=0.8, label="Eligible (CLIP ≥ 25.2, fidelity ≥ 75)")
        ax.scatter([f["objectives"][x] for f in finalists], [f["objectives"]["clip_score"] for f in finalists], s=46,
                   color=ORANGE, edgecolors="white", linewidths=1.5, label="Locked finalists", zorder=3)
        ax.axhline(26.215, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate("20-step DDIM reference CLIP", (ax.get_xlim()[0], 26.215), xytext=(4, 4), textcoords="offset points",
                    color=MUTED, fontsize=8)
        style(ax)
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
        ax.set_ylim(22, 27.6)
    axes[0].set_ylabel("CLIP prompt alignment (calibration)", color=MUTED, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(FIG / "search_candidates.png", dpi=200)
    print(f"figures -> {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
