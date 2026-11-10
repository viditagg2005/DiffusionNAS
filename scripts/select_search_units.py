#!/usr/bin/env python3
"""Choose the removable units that enter the v3 search space from the calibration probe.

Rule (fixed before any v3 search result exists): keep a unit if removing it
alone does not significantly lower CLIP prompt alignment (upper end of the
paired 95% bootstrap CI >= 0) and keeps CLIP image fidelity to the reference
>= 80. Combinations are left to the search, since removals interact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, default=ROOT / "artifacts" / "probe" / "probe_summary.json")
    parser.add_argument("--min-fidelity", type=float, default=80.0)
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "nas" / "units.json")
    args = parser.parse_args()

    rows = json.loads(args.probe.read_text(encoding="utf-8"))["rows"]
    kept, dropped = [], []
    for row in rows:
        if not row["label"].startswith("10_rm_"):
            continue
        unit = row["removed_units"]
        evidence = {"unit": unit, "clip_diff": round(row["clip_diff"], 3),
                    "clip_diff_ci": [round(row["clip_diff_ci_low"], 3), round(row["clip_diff_ci_high"], 3)],
                    "fidelity": round(row["fidelity"], 2), "black_images": row["black_images"],
                    "unet_weight_mb_saved": round(rows[0]["unet_weight_mb"] - row["unet_weight_mb"], 1),
                    "gflops_saved_pct": round(100 * (1 - row["gflops_per_image"] / rows[0]["gflops_per_image"]), 2)}
        (kept if row["clip_diff_ci_high"] >= 0 and row["fidelity"] >= args.min_fidelity else dropped).append(evidence)
    value = {
        "rule": f"single-unit removal: CLIP diff 95% CI upper bound >= 0 and CLIP image fidelity >= {args.min_fidelity}",
        "probe_reference_run": json.loads(args.probe.read_text(encoding="utf-8"))["reference_run"],
        "units": [item["unit"] for item in kept],
        "kept": kept,
        "dropped": dropped,
    }
    args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(f"kept {len(kept)} units: {value['units']}")
    print(f"dropped {len(dropped)}: {[item['unit'] for item in dropped]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
