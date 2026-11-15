#!/usr/bin/env python3
"""Pre-registered analysis of the SAEA benchmark (configs/nas/saea_benchmark.json). Written before any run.

Per cell (setting, problem, M, noise): two-sided Mann-Whitney U of sa_lhfid vs each other algorithm over seeds,
Holm over the comparisons in the cell; failed runs (timeout/error) are ranked worst. Counts wins/ties/losses
per comparison, separately for noise-free and noisy cells, and applies the protocol's claim rule.
Results: artifacts/analysis/saea_benchmark_<setting>.json only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas.stats import holm  # noqa: E402

PROTOCOL = json.loads((ROOT / "configs/nas/saea_benchmark.json").read_text())
INDICATORS = {"igd_plus": "min", "hv5_deliverable": "max"}
OURS = "sa_lhfid"


def value(row: dict, indicator: str) -> float:
    if row.get("status", "ok") != "ok":
        return np.inf if INDICATORS[indicator] == "min" else -np.inf  # ranked worst
    return row[indicator]


def main() -> int:
    from scipy.stats import mannwhitneyu

    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", default="main", choices=sorted(PROTOCOL["settings"]))
    args = parser.parse_args()
    rows = []
    for path in sorted((ROOT / "artifacts" / "saea_benchmark" / args.setting).glob("results_shard*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    latest = {}
    for row in rows:  # a resubmitted shard may re-run a failed task; keep the last record
        latest[(row["problem"], row["m"], row["noise"], row["algorithm"], row["seed"])] = row
    cells = defaultdict(lambda: defaultdict(list))
    for (problem, m, noise, algorithm, seed), row in latest.items():
        cells[(problem, m, noise)][algorithm].append(row)
    others = [a for a in PROTOCOL["algorithms"] if a != OURS]
    expected = PROTOCOL["settings"][args.setting]["seeds"]
    out = {"setting": args.setting, "cells": {}, "summary": {}, "failures": defaultdict(int), "incomplete_cells": []}
    tally = {ind: {noise: {o: {"wins": 0, "ties": 0, "losses": 0, "ratios": []} for o in others}
                   for noise in PROTOCOL["settings"][args.setting]["noise"]} for ind in INDICATORS}
    for (problem, m, noise), by in sorted(cells.items()):
        for algorithm, runs in by.items():
            out["failures"][algorithm] += sum(r.get("status", "ok") != "ok" for r in runs)
        if any(len(by.get(a, [])) < expected for a in PROTOCOL["algorithms"]):
            out["incomplete_cells"].append(f"{problem}_m{m}_n{noise}")
            continue
        cell = {}
        for indicator, direction in INDICATORS.items():
            ours = np.array([value(r, indicator) for r in by[OURS]])
            raw, entries = {}, {}
            for other in others:
                theirs = np.array([value(r, indicator) for r in by[other]])
                p = float(mannwhitneyu(ours, theirs, alternative="two-sided").pvalue)
                raw[other] = p
                med_ours, med_theirs = float(np.median(ours)), float(np.median(theirs))
                better = med_ours < med_theirs if direction == "min" else med_ours > med_theirs
                entries[other] = {"median_ours": med_ours, "median_other": med_theirs, "p": p, "ours_better_median": better}
            for other, p in holm(raw).items():
                e = entries[other]
                e["p_holm"] = p
                outcome = "ties" if p >= 0.05 else ("wins" if e["ours_better_median"] else "losses")
                e["outcome"] = outcome
                t = tally[indicator][noise][other]
                t[outcome] += 1
                if indicator == "igd_plus" and np.isfinite(e["median_ours"]) and np.isfinite(e["median_other"]) and e["median_other"] > 0:
                    t["ratios"].append(e["median_ours"] / e["median_other"])
            cell[indicator] = entries
        out["cells"][f"{problem}_m{m}_n{noise}"] = cell
    for indicator, by_noise in tally.items():
        for noise, by_other in by_noise.items():
            for other, t in by_other.items():
                ratio = statistics.median(t["ratios"]) if t["ratios"] else None
                verdict = None
                if indicator == "igd_plus":
                    verdict = ("better" if t["wins"] > t["losses"] and ratio is not None and ratio < 1
                               else "competitive" if t["losses"] <= t["wins"] else "worse")
                out["summary"].setdefault(indicator, {}).setdefault(str(noise), {})[other] = {
                    "wins": t["wins"], "ties": t["ties"], "losses": t["losses"], "median_igd_ratio": ratio, "claim": verdict}
    out["failures"] = dict(out["failures"])
    path = ROOT / "artifacts" / "analysis" / f"saea_benchmark_{args.setting}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    for indicator, by_noise in out["summary"].items():
        for noise, by_other in by_noise.items():
            print(f"{indicator} noise {noise}: " + "  ".join(
                f"{o} {v['wins']}/{v['ties']}/{v['losses']}" + (f" ({v['claim']})" if v["claim"] else "") for o, v in by_other.items()))
    if out["incomplete_cells"]:
        print(f"incomplete cells (skipped): {len(out['incomplete_cells'])}")
    print("failures:", out["failures"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
