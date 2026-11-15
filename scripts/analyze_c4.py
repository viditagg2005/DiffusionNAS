#!/usr/bin/env python3
"""Pre-registered C4 analysis (configs/nas/c4_protocol.json). Written before any C4 data existed.

Primary: HV(NFS) - HV(naive) within each arm on test3, paired prompt bootstrap, Holm over 4.
Secondary: HV(SA-LHFiD NFS) - HV(other arm NFS), Holm over 3. Results: artifacts/analysis/c4.json only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from c4_select import succeeded_runs  # noqa: E402
from diffusionnas import final_selection as fs  # noqa: E402
from diffusionnas.stats import holm, per_prompt_scores  # noqa: E402

DRAWS = 10_000
MARGIN = 0.5


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c4", type=Path, default=ROOT / "configs" / "generated" / "c4")
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts" / "c4" / "test" / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "analysis" / "c4.json")
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args()
    assignments = read(args.c4 / "assignments.json")
    runs = succeeded_runs(args.runs)
    labels = sorted({m for kind in ("nfs", "naive") for members in assignments[kind].values() for m in members})
    missing = [label for label in labels + ["ref_a", "ref_b", "ref_c"] if label not in runs]
    if missing:
        raise SystemExit(f"missing runs: {missing}")
    gpus = {read(runs[label] / "manifest.json")["provenance"]["gpu"]["name"] for label in labels + ["ref_a"]}
    if len(gpus) != 1:
        raise SystemExit(f"test ran on {sorted(gpus)}")

    prompts = sorted(per_prompt_scores(runs["ref_a"]))
    clip, fidelity = {}, {}
    for label in labels + ["ref_a"]:
        scores = per_prompt_scores(runs[label])
        clip[label] = np.array([scores[i] for i in prompts])
    for label in labels:
        scores = per_prompt_scores(runs[label], "clip_image_fidelity")
        fidelity[label] = np.array([scores[i] for i in prompts])
    summary = {label: read(runs[label] / "summary.json") for label in labels + ["ref_a", "ref_b", "ref_c"]}
    ref_latency = statistics.median(summary[r]["latency_ms"]["p50"] for r in ("ref_a", "ref_b", "ref_c"))
    ref = fs.box(float(clip["ref_a"].mean()), ref_latency)
    latency = {label: summary[label]["latency_ms"]["p50"] for label in labels}
    mb = {label: summary[label]["architecture"]["unet_weight_bytes"] / 2**20 for label in labels}

    def hv(members: list[str], index: np.ndarray) -> float:
        if not members:
            return 0.0
        points = np.array([[-clip[m][index].mean(), -fidelity[m][index].mean(), latency[m], mb[m]] for m in members])
        return fs.hypervolume(points, ref)

    sets = {f"{kind}:{arm}": members for kind in ("nfs", "naive") for arm, members in assignments[kind].items()}
    everything = np.arange(len(prompts))
    point = {name: hv(members, everything) for name, members in sets.items()}
    rng = np.random.default_rng(0)
    boot = {name: np.empty(args.draws) for name in sets}
    for b in range(args.draws):
        index = rng.integers(0, len(prompts), len(prompts))
        for name, members in sets.items():
            boot[name][b] = hv(members, index)

    def compare(a: str, b: str) -> dict:
        diff = boot[a] - boot[b]
        if np.all(diff == 0):
            p = 1.0
        else:
            p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()) + 1 / args.draws)
        return {"point_difference": point[a] - point[b], "median_bootstrap_difference": float(np.median(diff)),
                "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], "p": p,
                "identical_sets": sorted(sets[a]) == sorted(sets[b])}

    families = {"primary_nfs_vs_naive": {arm: compare(f"nfs:{arm}", f"naive:{arm}") for arm in assignments["nfs"]},
                "secondary_sa_lhfid_vs_arm": {arm: compare("nfs:sa_lhfid", f"nfs:{arm}")
                                              for arm in assignments["nfs"] if arm != "sa_lhfid"}}
    for tests in families.values():
        for key, p in holm({k: v["p"] for k, v in tests.items()}).items():
            tests[key]["p_holm"] = p
            tests[key]["claimed_better"] = bool(p < 0.05 and tests[key]["median_bootstrap_difference"] > 0)

    descriptive = {}
    for label in labels:
        diffs = list(clip[label] - clip["ref_a"])
        r = random.Random(0)
        means = sorted(statistics.fmean(r.choices(diffs, k=len(diffs))) for _ in range(args.draws))
        q = summary[label]["quality_metrics"]
        descriptive[label] = {
            "clip": float(clip[label].mean()), "clip_difference": float(np.mean(diffs)),
            "lower_97_5": means[int(0.025 * args.draws)], "fidelity": float(fidelity[label].mean()),
            "latency_p50_ms": latency[label], "latency_ratio": latency[label] / ref_latency, "unet_weight_mb": mb[label],
            "gflops_per_image": summary[label].get("gflops_per_image"), "peak_allocated_mb": summary[label]["peak_allocated_mb"],
            "black_images": q.get("black_images"), "kid_real_x1000": q.get("real_distribution", {}).get("kid", {}).get("mean_x1000"),
            "fid_real": q.get("real_distribution", {}).get("fid"),
            "in_box": bool(fs.in_box(np.array([-clip[label].mean(), -fidelity[label].mean(), latency[label], mb[label]]), ref)[0])}
    result = {"protocol": read(ROOT / "configs" / "nas" / "c4_protocol.json"), "box": ref.tolist(),
              "reference_latency_ms": {r: summary[r]["latency_ms"]["p50"] for r in ("ref_a", "ref_b", "ref_c")},
              "reference_clip": float(clip["ref_a"].mean()), "reference_black": summary["ref_a"]["quality_metrics"]["black_images"],
              "hypervolume": point, "families": families, "descriptive": descriptive, "gpu": gpus.pop()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for name, value in point.items():
        print(f"{name:16s} HV {value:.4g}")
    for family, tests in families.items():
        print(family)
        for key, t in tests.items():
            print(f"  {key:9s} diff {t['point_difference']:+.4g} 95% CI [{t['ci95'][0]:+.4g}, {t['ci95'][1]:+.4g}] "
                  f"pHolm {t['p_holm']:.4f} claimed {t['claimed_better']}{' (identical sets)' if t['identical_sets'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
