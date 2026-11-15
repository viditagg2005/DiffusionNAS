#!/usr/bin/env python3
"""Regenerate every derived statistic quoted in the report that is not produced by another script.

Writes artifacts/analysis/derived_stats.json:
- CLIP split-half reliability among in-box calibration candidates (v3 searches),
- in-box rates per arm across all 6 seeds of C2 (random / NSGA-II seeds 1-3 from artifacts/nas),
- RVEA and MOEA/D vs LHFiD on the surrogate benchmark (labelled extension),
- LHFiD nadir-estimation firing counts on the surrogate benchmark,
- prompt-category routing vs the best single policy and vs the best two-policy mixture.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BOX = {"clip_score": 25.2, "clip_image_fidelity": 75.0, "latency_p50_ms": 600.0, "unet_weight_mb": 1700.0}


def in_box(o: dict) -> bool:
    return (o["clip_score"] >= BOX["clip_score"] and o["clip_image_fidelity"] >= BOX["clip_image_fidelity"]
            and o["latency_p50_ms"] <= BOX["latency_p50_ms"] and o["unet_weight_mb"] <= BOX["unet_weight_mb"])


def clip_reliability() -> dict:
    from scipy.stats import spearmanr

    halves = ([], [])
    for seed_dir in sorted((ROOT / "artifacts/nas").glob("seed*")):
        for arm in ("random", "nsga2"):
            for candidate in json.loads((seed_dir / arm / "search_results.json").read_text())["candidates"]:
                if not in_box(candidate["objectives"]):
                    continue
                run = seed_dir / arm / "runs" / Path(candidate["run_dir"]).name
                rows = [json.loads(line) for line in (run / "quality_measurements.jsonl").read_text().splitlines() if line]
                halves[0].append(statistics.fmean(r["clip_score"] for r in rows if r["prompt_index"] % 2 == 0))
                halves[1].append(statistics.fmean(r["clip_score"] for r in rows if r["prompt_index"] % 2 == 1))
    r = float(spearmanr(*halves).statistic)
    full = 2 * r / (1 + r)
    return {"in_box_candidates": len(halves[0]), "split_half_spearman": r, "spearman_brown": full, "ceiling": full ** 0.5}


def in_box_rates() -> dict:
    out = {}
    for arm in ("sa_lhfid", "sa_nsga2", "random", "nsga2"):
        after, total_after = 0, 0
        for seed in range(1, 7):
            path = ROOT / "artifacts/nas_c2" / f"seed{seed}" / arm / "search_results.json"
            if not path.exists():
                path = ROOT / "artifacts/nas" / f"seed{seed}" / arm / "search_results.json"
            candidates = json.loads(path.read_text())["candidates"]
            later = candidates[50:]
            after += sum(in_box(c["objectives"]) for c in later)
            total_after += len(later)
        out[arm] = {"in_box_after_first_50": after, "evaluations_after_first_50": total_after, "rate": after / total_after}
    return out


def extension_comparison() -> dict:
    from scipy.stats import wilcoxon

    runs = (json.loads((ROOT / "artifacts/surrogate_benchmark/results.json").read_text())["runs"]
            + json.loads((ROOT / "artifacts/surrogate_benchmark/results_extension_rvea_moead.json").read_text())["runs"])
    out = {}
    for key in ("M4", "M6"):
        for indicator, get in (("archive_hv@3360", lambda r: r["archive_hv"]["3360"]), ("final_population_hv", lambda r: r["final_population_hv"])):
            values = {(r["algorithm"], r["surrogate"], r["seed"]): get(r) for r in runs if r["objective_set"] == key and get(r) is not None}
            cells = sorted({(s, d) for _, s, d in values})
            block = {"medians": {a: statistics.median(values[(a, s, d)] for s, d in cells)
                                 for a in ("nsga2", "nsga3", "moead", "rvea", "lhfid") if (a, *cells[0]) in values}}
            for other in ("rvea", "moead", "nsga3"):
                diffs = [values[("lhfid", s, d)] - values[(other, s, d)] for s, d in cells]
                signs = {float(np.sign(statistics.median(values[("lhfid", s, d)] - values[(other, s, d)] for s2, d in cells if s2 == s)))
                         for s in sorted({c[0] for c in cells})}
                block[f"lhfid_minus_{other}"] = {"median_difference": statistics.median(diffs), "p": float(wilcoxon(diffs).pvalue),
                                                 "consistent_sign": len(signs) == 1}
            out[f"{key} {indicator}"] = block
    return out


def nadir_firing() -> dict:
    runs = json.loads((ROOT / "artifacts/surrogate_benchmark/results.json").read_text())["runs"]
    lhfid = [r for r in runs if r["algorithm"].startswith("lhfid")]
    firsts = [r["nadir_generations"][0] for r in lhfid if r["nadir_generations"]]
    return {"runs": len(lhfid), "fired": len(firsts), "median_first_generation": statistics.median(firsts)}


def routing_vs_mixture() -> dict:
    from routing_feasibility import BUDGETS, load

    strata, runs = load()
    rng = np.random.default_rng(1)
    names = sorted(set(strata))
    idx = {s: np.array([i for i, x in enumerate(strata) if x == s]) for s in names}
    w = np.array([len(idx[s]) / len(strata) for s in names])
    out = {}
    for portfolio, labels in (("fp16_no_removal", sorted(k for k, r in runs.items() if r["precision"] == "fp16" and not r["removed"])),
                              ("int8_no_removal", sorted(k for k, r in runs.items() if r["precision"] == "int8" and not r["removed"]))):
        lat = np.array([runs[k]["latency"] for k in labels])
        clip = np.stack([runs[k]["clip"].mean(axis=0) for k in labels])
        gains = {b: [] for b in BUDGETS}
        for _ in range(200):
            sel, ev = {}, {}
            for s in names:
                order = rng.permutation(idx[s])
                sel[s], ev[s] = order[: len(order) // 2], order[len(order) // 2:]
            ss = np.stack([clip[:, sel[s]].mean(1) for s in names], 1)
            es = np.stack([clip[:, ev[s]].mean(1) for s in names], 1)
            sg, eg = ss @ w, es @ w
            total = w[0] * ss[:, 0][:, None, None] + w[1] * ss[:, 1][None, :, None] + w[2] * ss[:, 2][None, None, :]
            mean_lat = w[0] * lat[:, None, None] + w[1] * lat[None, :, None] + w[2] * lat[None, None, :]
            for budget in BUDGETS:
                if not (lat <= budget).any():
                    continue
                m = np.where(mean_lat <= budget, total, -np.inf)
                a, b, c = np.unravel_index(np.argmax(m), m.shape)
                routed = w[0] * es[a, 0] + w[1] * es[b, 1] + w[2] * es[c, 2]
                best = (-np.inf, None)
                for i in range(len(lat)):
                    for j in range(len(lat)):
                        if lat[i] <= budget and (lat[j] <= budget or lat[j] != lat[i]):
                            lam = 1.0 if lat[j] <= budget else (lat[j] - budget) / (lat[j] - lat[i])
                            if 0 <= lam <= 1 and lam * sg[i] + (1 - lam) * sg[j] > best[0]:
                                best = (lam * sg[i] + (1 - lam) * sg[j], (i, j, lam))
                i, j, lam = best[1]
                gains[budget].append(float(routed - (lam * eg[i] + (1 - lam) * eg[j])))
        out[portfolio] = {str(b): {"median_gain_over_mixture": float(np.median(g)), "share_positive": float(np.mean(np.array(g) > 0)),
                                   "p5": float(np.percentile(g, 5)), "p95": float(np.percentile(g, 95))} for b, g in gains.items() if g}
    return out


def main() -> int:
    result = {"clip_reliability": clip_reliability(), "in_box_rates_c2": in_box_rates(),
              "extension_rvea_moead_vs_lhfid": extension_comparison(), "lhfid_nadir_firing": nadir_firing(),
              "routing_vs_best_mixture": routing_vs_mixture()}
    out = ROOT / "artifacts/analysis/derived_stats.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"clip_reliability": result["clip_reliability"], "in_box_rates_c2": result["in_box_rates_c2"],
                      "lhfid_nadir_firing": result["lhfid_nadir_firing"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
