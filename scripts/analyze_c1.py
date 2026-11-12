#!/usr/bin/env python3
"""Pre-registered C1 analysis (configs/nas/c1_protocol.json). Written before any C1 data existed."""

from __future__ import annotations

import json
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas.stats import holm, per_prompt_scores  # noqa: E402

PROTOCOL = json.loads((ROOT / "configs/nas/c1_protocol.json").read_text())


def bootstrap_means(a: dict, b: dict, draws: int = 10_000, seed: int = 0) -> tuple[float, list[float]]:
    keys = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    return statistics.fmean(diffs), sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(draws))


def main() -> int:
    runs = {}
    for manifest in (ROOT / "artifacts/c1/runs").glob("*/manifest.json"):
        if json.loads(manifest.read_text()).get("status") == "succeeded":
            runs[json.loads((manifest.parent / "config.json").read_text())["label"]] = manifest.parent
    clip = {label: per_prompt_scores(run) for label, run in runs.items()}
    margin = PROTOCOL["margin_clip"]
    tests = {}
    for name, label in (("H1", "nas_04"), ("H2", "nas_01"), ("H3", "int8_ddim20")):
        mean, means = bootstrap_means(clip[label], clip["reference"])
        tests[name] = {"comparison": f"{label} - reference", "mean_difference": mean, "lower_97_5": means[250],
                       "p": (sum(m <= -margin for m in means) + 1) / (len(means) + 1), "type": f"non-inferiority, margin {margin}"}
    for name, label in (("H4", "bksdm_small_ddim10_int8"), ("H5", "bksdm_base_ddim10_int8")):
        mean, means = bootstrap_means(clip["nas_04"], clip[label])
        tail = min(sum(m <= 0 for m in means), sum(m >= 0 for m in means))
        tests[name] = {"comparison": f"nas_04 - {label}", "mean_difference": mean, "ci95": [means[250], means[9749]],
                       "p": min(1.0, 2 * (tail + 1) / (len(means) + 1)), "type": "superiority, two-sided"}
    for name, p in holm({k: v["p"] for k, v in tests.items()}).items():
        tests[name]["p_holm"] = p
        tests[name]["supported"] = p < 0.05
    secondary = {}
    for label, run in runs.items():
        summary = json.loads((run / "summary.json").read_text())
        q = summary["quality_metrics"]
        secondary[label] = {
            "latency_p50_ms": summary["latency_ms"]["p50"], "gflops_per_image": summary.get("gflops_per_image"),
            "peak_allocated_mb": summary["peak_allocated_mb"], "unet_weight_mb": summary["architecture"]["unet_weight_bytes"] / 2**20,
            "unet_parameters_m": summary["architecture"]["unet_parameters"] / 1e6, "clip_score": q["clip_score"]["mean"],
            "clip_image_fidelity": q.get("clip_image_fidelity", {}).get("mean"), "black_images": q.get("black_images"),
            "kid_relative_x1000": q.get("relative_distribution", {}).get("kid", {}).get("mean_x1000"),
            "kid_real_x1000": q.get("real_distribution", {}).get("kid", {}).get("mean_x1000"),
            "fid_real": q.get("real_distribution", {}).get("fid"),
        }
    result = {"protocol": PROTOCOL, "tests": tests, "secondary": secondary}
    out = ROOT / "artifacts/analysis/c1.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for name, t in tests.items():
        print(f"{name} {t['comparison']:38s} {t['mean_difference']:+.3f}  pHolm {t['p_holm']:.4f}  supported {t['supported']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
