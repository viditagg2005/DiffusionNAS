#!/usr/bin/env python3
"""Pre-registered C3 analysis (configs/nas/c3_protocol.json). Written before any C3 data existed.

Each search arm contributes its locked set (configs/generated/c3/assignments.json). Primary: 4-D
hypervolume of each arm's set on test2, compared by a paired prompt bootstrap with Holm correction.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas.stats import holm, per_prompt_scores  # noqa: E402

PROTOCOL = json.loads((ROOT / "configs/nas/c3_protocol.json").read_text())
ASSIGNMENTS = json.loads((ROOT / "configs/generated/c3/assignments.json").read_text())


def main() -> int:
    import moocore

    runs = {}
    for manifest in (ROOT / "artifacts/c3/runs").glob("*/manifest.json"):
        if json.loads(manifest.read_text()).get("status") == "succeeded":
            runs[json.loads((manifest.parent / "config.json").read_text())["label"]] = manifest.parent
    labels = sorted({label for members in ASSIGNMENTS["arms"].values() for label in members})
    missing = [label for label in labels + ["ref_a"] if label not in runs]
    if missing:
        raise SystemExit(f"missing runs: {missing}")
    prompts = sorted(per_prompt_scores(runs["ref_a"]))
    clip = {label: np.array([per_prompt_scores(runs[label])[i] for i in prompts]) for label in labels + ["ref_a"]}
    fidelity = {label: np.array([per_prompt_scores(runs[label], "clip_image_fidelity")[i] for i in prompts]) for label in labels}
    summary = {label: json.loads((runs[label] / "summary.json").read_text()) for label in labels + ["ref_a"]}
    latency = {label: summary[label]["latency_ms"]["p50"] for label in labels}
    weights = {label: summary[label]["architecture"]["unet_weight_bytes"] / 2**20 for label in labels}
    clip_floor = float(clip["ref_a"].mean()) - 1.0
    ref = np.array([-clip_floor, -75.0, 600.0, 1700.0])

    def hypervolume(members, index):
        f = np.array([[-clip[m][index].mean(), -fidelity[m][index].mean(), latency[m], weights[m]] for m in members])
        inside = f[(f < ref).all(axis=1)]
        return float(moocore.hypervolume(inside, ref=ref)) if len(inside) else 0.0

    everything = np.arange(len(prompts))
    point = {arm: hypervolume(members, everything) for arm, members in ASSIGNMENTS["arms"].items()}
    rng = np.random.default_rng(0)
    draws = 10_000
    boot = {arm: np.empty(draws) for arm in ASSIGNMENTS["arms"]}
    for b in range(draws):
        index = rng.integers(0, len(prompts), len(prompts))
        for arm, members in ASSIGNMENTS["arms"].items():
            boot[arm][b] = hypervolume(members, index)
    tests, raw = {}, {}
    for other in ("sa_nsga2", "random", "nsga2"):
        diff = boot["sa_lhfid"] - boot[other]
        p = min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()) + 1 / draws)
        raw[other] = p
        tests[f"sa_lhfid vs {other}"] = {"point_difference": point["sa_lhfid"] - point[other],
                                         "median_bootstrap_difference": float(np.median(diff)),
                                         "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], "p": p}
    for other, p in holm(raw).items():
        entry = tests[f"sa_lhfid vs {other}"]
        entry["p_holm"] = p
        entry["claimed_better"] = bool(p < 0.05 and entry["median_bootstrap_difference"] > 0)

    secondary = {}
    margin = 0.5
    for arm, members in ASSIGNMENTS["arms"].items():
        raw_ni = {}
        for label in members:
            diffs = list(clip[label] - clip["ref_a"])
            r = random.Random(0)
            means = sorted(statistics.fmean(r.choices(diffs, k=len(diffs))) for _ in range(draws))
            raw_ni[label] = (sum(m <= -margin for m in means) + 1) / (draws + 1)
            q = summary[label]["quality_metrics"]
            secondary.setdefault(arm, {})[label] = {
                "clip": float(clip[label].mean()), "clip_difference": float(np.mean(diffs)), "lower_97_5": means[250],
                "fidelity": float(fidelity[label].mean()), "latency_p50_ms": latency[label], "unet_weight_mb": weights[label],
                "gflops_per_image": summary[label].get("gflops_per_image"), "peak_allocated_mb": summary[label]["peak_allocated_mb"],
                "black_images": q.get("black_images"), "kid_real_x1000": q.get("real_distribution", {}).get("kid", {}).get("mean_x1000"),
                "fid_real": q.get("real_distribution", {}).get("fid")}
        for label, p in holm(raw_ni).items():
            secondary[arm][label]["noninferior_p_holm_within_arm"] = p
    result = {"protocol": PROTOCOL, "clip_floor": clip_floor, "hypervolume": point, "tests": tests, "secondary": secondary,
              "reference_latency_ms": {k: json.loads((runs[k] / "summary.json").read_text())["latency_ms"]["p50"]
                                       for k in ("ref_a", "ref_b", "ref_c") if k in runs}}
    out = ROOT / "artifacts/analysis/c3.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for arm, value in point.items():
        print(f"{arm:9s} HV {value:.4g}")
    for name, entry in tests.items():
        print(f"{name:22s} diff {entry['point_difference']:+.4g}  95% CI [{entry['ci95'][0]:+.4g}, {entry['ci95'][1]:+.4g}]  "
              f"pHolm {entry['p_holm']:.4f}  claimed {entry['claimed_better']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
