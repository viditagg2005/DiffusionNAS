#!/usr/bin/env python3
"""Compare surrogate predictions with real calibration measurements for the confirmation points."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas.nas_search import read_nas_objectives  # noqa: E402


def main() -> int:
    points = {p["label"]: p for p in json.loads((ROOT / "artifacts/surrogate_benchmark/confirmation_points.json").read_text())}
    runs = {}
    for manifest in (ROOT / "artifacts/confirmation/runs").glob("*/manifest.json"):
        if json.loads(manifest.read_text()).get("status") == "succeeded":
            runs[json.loads((manifest.parent / "config.json").read_text())["label"]] = manifest.parent
    reference = read_nas_objectives(runs["reference"])
    pooled = json.loads((ROOT / "artifacts/nas/finalists.json").read_text())["pooled_reference_latency_ms"]
    rows, errors = [], {}
    for label, point in points.items():
        if label not in runs:
            continue
        measured = read_nas_objectives(runs[label])
        measured["latency_p50_ms"] *= pooled / reference["latency_p50_ms"]  # same normalization as training data
        predicted = {k: (-v if k in ("clip_score", "clip_image_fidelity") else v) for k, v in point["predicted_minimization"].items()}
        row = {"label": label, "algorithm": point["algorithm"], "measured": measured, "predicted": predicted}
        rows.append(row)
        for name in predicted:
            errors.setdefault(name, []).append(measured[name] - predicted[name])
    summary = {name: {"mean_error": statistics.fmean(v), "mean_abs_error": statistics.fmean(abs(x) for x in v)}
               for name, v in errors.items()}
    in_box = lambda m: (m["clip_score"] >= 25.2 and m["clip_image_fidelity"] >= 75 and m["latency_p50_ms"] <= 600
                        and m["unet_weight_mb"] <= 1700)
    by_algorithm = {}
    for row in rows:
        stats = by_algorithm.setdefault(row["algorithm"], {"points": 0, "inside_reference_box": 0})
        stats["points"] += 1
        stats["inside_reference_box"] += int(in_box(row["measured"]))
    result = {"errors": summary, "by_algorithm": by_algorithm, "rows": rows}
    out = ROOT / "artifacts/analysis/confirmation.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    for name, s in summary.items():
        print(f"{name:22s} bias {s['mean_error']:+9.2f}  MAE {s['mean_abs_error']:9.2f}")
    print(by_algorithm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
