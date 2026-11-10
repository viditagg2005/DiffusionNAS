#!/usr/bin/env python3
"""Tabulate the sensitivity probe: cost saved versus quality lost for each removed unit or group."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas.stats import paired_bootstrap, per_prompt_scores  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {}
    for manifest in sorted(args.runs_root.glob("*/manifest.json")):
        if read(manifest).get("status") != "succeeded":
            continue
        config = read(manifest.parent / "config.json")
        runs[config["label"]] = manifest.parent  # latest succeeded run per label wins (sorted by timestamp)
    if "00_reference" not in runs:
        raise SystemExit("reference run missing")
    reference = runs["00_reference"]
    ref_summary = read(reference / "summary.json")
    ref_clip = per_prompt_scores(reference)

    rows = []
    for label, run in sorted(runs.items()):
        summary = read(run / "summary.json")
        config = read(run / "config.json")
        arch = summary["architecture"]
        quality = summary["quality_metrics"]
        clip_diff = paired_bootstrap(per_prompt_scores(run), ref_clip) if run != reference else None
        rows.append({
            "label": label,
            "removed_units": " ".join((config.get("architecture") or {}).get("removed_units", [])),
            "weight_precision": arch["weight_precision"],
            "unet_params_m": arch["unet_parameters"] / 1e6,
            "unet_weight_mb": arch["unet_weight_bytes"] / 2**20,
            "gflops_per_image": summary.get("gflops_per_image"),
            "latency_p50_ms": summary["latency_ms"]["p50"],
            "speedup": ref_summary["latency_ms"]["p50"] / summary["latency_ms"]["p50"],
            "peak_allocated_mb": summary["peak_allocated_mb"],
            "clip_score": quality["clip_score"]["mean"],
            "clip_diff": clip_diff["mean_difference"] if clip_diff else 0.0,
            "clip_diff_ci_low": clip_diff["ci95_low"] if clip_diff else 0.0,
            "clip_diff_ci_high": clip_diff["ci95_high"] if clip_diff else 0.0,
            "fidelity": quality["clip_image_fidelity"]["mean"],
            "black_images": quality["black_images"],
            "gpu": read(run / "manifest.json")["provenance"]["gpu"]["name"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"reference_run": reference.name, "rows": rows}, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'label':28s} {'MB':>7s} {'GFLOP':>7s} {'ms':>6s} {'dCLIP [95% CI]':>24s} {'fid':>6s} blk")
    for r in rows:
        gflops = f"{r['gflops_per_image']:.0f}" if r["gflops_per_image"] else "-"
        print(f"{r['label']:28s} {r['unet_weight_mb']:7.0f} {gflops:>7s} {r['latency_p50_ms']:6.0f} "
              f"{r['clip_diff']:+6.2f} [{r['clip_diff_ci_low']:+.2f},{r['clip_diff_ci_high']:+.2f}] {r['fidelity']:6.2f} {r['black_images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
