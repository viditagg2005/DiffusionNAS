#!/usr/bin/env python3
"""Pre-registered C2 analysis (configs/nas/c2_protocol.json): real SA-LHFiD vs SA-NSGA-II, random, NSGA-II.

Written before any C2 data existed. Primary: 4-D hypervolume (CLIP, fidelity, normalized latency,
U-Net MB). Robustness: GFLOPs in place of latency (drift-free); 3-D without CLIP.
Latency is normalized by each job's own reference times the median reference over all jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas.stats import holm  # noqa: E402

PROTOCOL = json.loads((ROOT / "configs/nas/c2_protocol.json").read_text())
INDICATORS = {
    "primary_4d": [("clip_score", "max", 25.2), ("clip_image_fidelity", "max", 75.0),
                   ("latency_p50_ms", "min", 600.0), ("unet_weight_mb", "min", 1700.0)],
    "gflops_4d": [("clip_score", "max", 25.2), ("clip_image_fidelity", "max", 75.0),
                  ("gflops_per_image", "min", 36000.0), ("unet_weight_mb", "min", 1700.0)],
    "no_clip_3d": [("clip_image_fidelity", "max", 75.0), ("latency_p50_ms", "min", 600.0), ("unet_weight_mb", "min", 1700.0)],
}


def hypervolume(objectives: list[dict], spec) -> float:
    import moocore

    f = np.array([[-o[n] if d == "max" else o[n] for n, d, _ in spec] for o in objectives])
    ref = np.array([-v if d == "max" else v for _, d, v in spec])
    inside = f[(f < ref).all(axis=1)]
    return float(moocore.hypervolume(inside, ref=ref)) if len(inside) else 0.0


def collect() -> list[dict]:
    """Every run of every arm/seed with its job reference latency."""
    runs = []
    sources = [(ROOT / "artifacts/nas", ("random", "nsga2"), (1, 2, 3)),
               (ROOT / "artifacts/nas_c2", tuple(PROTOCOL["arms"]), (1, 2, 3, 4, 5, 6))]
    for root, arms, seeds in sources:
        for seed in seeds:
            seed_dir = root / f"seed{seed}"
            references = list((seed_dir / "reference" / "runs").glob("*")) if seed_dir.exists() else []
            if not references:
                continue
            reference = json.loads((references[0] / "summary.json").read_text())["latency_ms"]["p50"]
            for arm in arms:
                path = seed_dir / arm / "search_results.json"
                if path.exists() and json.loads(path.read_text())["status"] == "succeeded":
                    results = json.loads(path.read_text())
                    runs.append({"arm": arm, "seed": seed, "source": root.name, "reference_ms": reference,
                                 "candidates": [c["objectives"] for c in results["candidates"]]})
    return runs


def main() -> int:
    from scipy.stats import mannwhitneyu

    runs = collect()
    pooled = statistics.median({(r["source"], r["seed"]): r["reference_ms"] for r in runs}.values())
    for run in runs:
        for objective in run["candidates"]:
            objective["latency_p50_ms"] = objective["latency_p50_ms"] * pooled / run["reference_ms"]
        run["hypervolume"] = {name: hypervolume(run["candidates"], spec) for name, spec in INDICATORS.items()}
    arms = PROTOCOL["arms"]
    by_arm = {a: [r for r in runs if r["arm"] == a] for a in arms}
    summary = {a: {name: {"n": len(rs), "median": statistics.median(r["hypervolume"][name] for r in rs) if rs else None}
                   for name in INDICATORS} for a, rs in by_arm.items()}
    tests, raw = {}, {}
    for other in ("sa_nsga2", "random", "nsga2"):
        a, b = by_arm["sa_lhfid"], by_arm[other]
        if len(a) < 2 or len(b) < 2:
            continue
        entry = {}
        for name in INDICATORS:
            x, y = [r["hypervolume"][name] for r in a], [r["hypervolume"][name] for r in b]
            entry[name] = {"median_difference": statistics.median(x) - statistics.median(y),
                           "p": float(mannwhitneyu(x, y, alternative="two-sided").pvalue)}
        raw[other] = entry["primary_4d"]["p"]
        tests[f"sa_lhfid vs {other}"] = entry
    for other, p in holm(raw).items():
        entry = tests[f"sa_lhfid vs {other}"]
        entry["primary_p_holm"] = p
        signs = {np.sign(entry[name]["median_difference"]) for name in INDICATORS}
        entry["claimed_better"] = bool(p < 0.05 and signs == {1.0})
    result = {"protocol": PROTOCOL, "pooled_reference_ms": pooled, "summary": summary, "tests": tests,
              "runs": [{k: v for k, v in r.items() if k != "candidates"} for r in runs]}
    out = ROOT / "artifacts/analysis/c2.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for arm, block in summary.items():
        print(f"{arm:9s} n={block['primary_4d']['n']}  " + "  ".join(
            f"{name} {block[name]['median']:.4g}" if block[name]["median"] is not None else f"{name} -" for name in INDICATORS))
    for name, entry in tests.items():
        print(f"{name:22s} primary dmed {entry['primary_4d']['median_difference']:+.4g}  pHolm {entry.get('primary_p_holm', float('nan')):.3g}  "
              f"claimed {entry.get('claimed_better')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
