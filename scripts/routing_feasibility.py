#!/usr/bin/env python3
"""Exploratory routing checks on the validation runs (prompts already seen; confirm on a fresh split).

1. Held-out prompts: per stratum, select on a random half of prompts (both seeds),
   evaluate on the other half; 200 splits; gain of the stratum router over the best
   static policy at matched mean latency.
2. Single-weight portfolios: fp16 without removal, int8 without removal.
3. Per-prompt sensitivity reliability: correlation of per-prompt CLIP drop vs the
   reference between seeds 303 and 404.
"""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (150, 200, 250, 300, 400, 500)


def load():
    strata = [p["stratum"] for p in json.loads((ROOT / "data/prompts/validation.json").read_text())["prompts"]]
    runs = {}
    for manifest in (ROOT / "artifacts/validation_v3/runs").glob("*/manifest.json"):
        config = json.loads((manifest.parent / "config.json").read_text())
        if not config["model"]["id"].startswith("stable-diffusion") or config["label"] in ("ref_b", "ref_c"):
            continue
        summary = json.loads((manifest.parent / "summary.json").read_text())
        clip = np.zeros((2, 128))
        for row in map(json.loads, (manifest.parent / "quality_measurements.jsonl").read_text().splitlines()):
            clip[0 if row["seed"] == 303 else 1, row["prompt_index"]] = row["clip_score"]
        arch = config.get("architecture") or {}
        runs[config["label"]] = {"latency": summary["latency_ms"]["p50"], "clip": clip,
                                 "removed": bool(arch.get("removed_units")), "precision": arch.get("weight_precision", "fp16")}
    return strata, runs


def router_gain(labels, runs, strata, rng, splits=200):
    names = sorted(set(strata))
    idx = {s: np.array([i for i, x in enumerate(strata) if x == s]) for s in names}
    weights = np.array([len(idx[s]) / len(strata) for s in names])
    latency = np.array([runs[k]["latency"] for k in labels])
    clip = np.stack([runs[k]["clip"].mean(axis=0) for k in labels])  # policy x prompt, mean over seeds
    gains = {b: [] for b in BUDGETS}
    for _ in range(splits):
        sel, ev = {}, {}
        for s in names:
            order = rng.permutation(idx[s])
            half = len(order) // 2
            sel[s], ev[s] = order[:half], order[half:]
        sel_score = np.stack([clip[:, sel[s]].mean(axis=1) for s in names], axis=1)  # P x 3
        ev_score = np.stack([clip[:, ev[s]].mean(axis=1) for s in names], axis=1)
        total = (weights[0] * sel_score[:, 0][:, None, None] + weights[1] * sel_score[:, 1][None, :, None]
                 + weights[2] * sel_score[:, 2][None, None, :])
        mean_lat = (weights[0] * latency[:, None, None] + weights[1] * latency[None, :, None] + weights[2] * latency[None, None, :])
        static_sel = sel_score @ weights
        static_ev = ev_score @ weights
        for budget in BUDGETS:
            ok = latency <= budget
            if not ok.any():
                continue
            static = np.where(ok)[0][np.argmax(static_sel[ok])]
            masked = np.where(mean_lat <= budget, total, -np.inf)
            a, b, c = np.unravel_index(np.argmax(masked), masked.shape)
            routed = weights[0] * ev_score[a, 0] + weights[1] * ev_score[b, 1] + weights[2] * ev_score[c, 2]
            gains[budget].append(routed - static_ev[static])
    return {b: (float(np.median(g)), float(np.mean(np.array(g) > 0)), float(np.percentile(g, 5)), float(np.percentile(g, 95)))
            for b, g in gains.items() if g}


def main() -> int:
    strata, runs = load()
    rng = np.random.default_rng(0)
    portfolios = {
        "all SD1.5 (mixed weights)": sorted(runs),
        "fp16, no removal": sorted(k for k, r in runs.items() if r["precision"] == "fp16" and not r["removed"]),
        "int8, no removal": sorted(k for k, r in runs.items() if r["precision"] == "int8" and not r["removed"]),
    }
    report = {}
    for name, labels in portfolios.items():
        result = router_gain(labels, runs, strata, rng)
        report[name] = {"policies": labels, "gain_by_budget": result}
        print(f"\n{name} ({len(labels)} policies; latencies {min(runs[k]['latency'] for k in labels):.0f}-{max(runs[k]['latency'] for k in labels):.0f} ms)")
        for budget, (median, share, lo, hi) in result.items():
            print(f"  budget {budget}: median gain {median:+.3f}  P(gain>0) {share:.2f}  90% range [{lo:+.2f}, {hi:+.2f}]")
    reference = runs["ref_a"]["clip"]
    correlations = []
    for label, run in runs.items():
        if label == "ref_a":
            continue
        drop = run["clip"] - reference
        correlations.append(float(np.corrcoef(drop[0], drop[1])[0, 1]))
    report["per_prompt_drop_seed_correlation"] = {"median": float(np.median(correlations)), "min": min(correlations), "max": max(correlations)}
    print("\nper-prompt CLIP-drop correlation between seeds:", {k: round(v, 3) for k, v in report["per_prompt_drop_seed_correlation"].items()})
    (ROOT / "artifacts/analysis/routing_feasibility.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
