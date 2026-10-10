#!/usr/bin/env python3
"""Lock validation finalists from calibration results with a pre-registered rule.

Committed before the v2 search and control results exist, so the rule cannot
be tuned to them:

1. Pool every succeeded, CLIP-scored calibration run under the given roots.
   All runs must report the same GPU model.
2. Merge runs with the same execution signature (identical images); use the
   median of their p50 latencies and the maximum peak memory.
3. Drop policies whose mean CLIP is more than ``--max-clip-drop`` below the
   pool's 20-step DDIM guidance-7.5 baseline.
4. Keep the 3-objective nondominated set (CLIP max, p50 latency min, peak memory min).
5. If more than ``--max-finalists`` remain, keep the fastest and the
   highest-CLIP policy and fill the rest at evenly spaced latency ranks.
6. Always add the reference comparators (20-step DDIM, DeepCache i3 b0,
   native 4-step DPM-Solver++) whether or not they survive.

The output is a front file that ``diffusionnas holdout`` accepts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas.pareto import BASE_OBJECTIVES, nondominated  # noqa: E402
from diffusionnas.policy_space import config_signature  # noqa: E402

REFERENCES = {
    "baseline_ddim20": {"method": "baseline", "scheduler": "ddim", "steps": 20, "guidance": 7.5},
    "deepcache_i3_b0": {"method": "deepcache", "scheduler": "ddim", "steps": 20, "guidance": 7.5},
    "native_dpm4": {"method": "baseline", "scheduler": "dpm_solver++", "steps": 4, "guidance": 7.5},
}


def reference_name(config: dict[str, Any]) -> str | None:
    inference = config["inference"]
    for name, spec in REFERENCES.items():
        if (
            config["method"] == spec["method"]
            and inference["scheduler"] == spec["scheduler"]
            and inference["num_inference_steps"] == spec["steps"]
            and inference["guidance_scale"] == spec["guidance"]
            and "timesteps" not in inference
            and (spec["method"] != "deepcache" or (config["deepcache"]["interval"], config["deepcache"]["branch_id"]) == (3, 0))
        ):
            return name
    return None


def load_pool(roots: list[Path], prompts: list[str], seeds: list[int]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    gpus: set[str] = set()
    for root in roots:
        for manifest_path in sorted(root.glob("**/manifest.json")):
            run_dir = manifest_path.parent
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "succeeded":
                continue
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            clip = summary.get("quality_metrics", {}).get("clip_score")
            if clip is None or config["prompts"] != prompts or config["seeds"] != seeds:
                continue
            gpus.add(manifest["provenance"]["gpu"]["name"])
            groups.setdefault(config_signature(config), []).append(
                {"run_dir": run_dir, "run_id": manifest["run_id"], "method": manifest["method"],
                 "fidelity": manifest["implementation_fidelity"], "config": config, "summary": summary}
            )
    if len(gpus) > 1:
        raise ValueError(f"pool mixes GPU models {sorted(gpus)}; latencies are not comparable")
    pool = {}
    for signature, runs in groups.items():
        first = runs[0]
        pool[signature] = {
            "run_id": first["run_id"],
            "run_dir": str(first["run_dir"].resolve()),
            "method": first["method"],
            "implementation_fidelity": first["fidelity"],
            "reference": reference_name(first["config"]),
            "duplicate_runs": len(runs),
            "policy": {k: first["config"].get(k) for k in ("method", "inference", "deepcache", "flexiffusion")},
            "objectives": {
                "clip_score": first["summary"]["quality_metrics"]["clip_score"]["mean"],
                "latency_p50_ms": statistics.median(r["summary"]["latency_ms"]["p50"] for r in runs),
                "peak_allocated_mb": max(r["summary"]["peak_allocated_mb"] for r in runs),
            },
        }
    return pool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompts" / "calibration.json")
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--max-clip-drop", type=float, default=1.0)
    parser.add_argument("--max-finalists", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompts = [item["prompt"] for item in json.loads(args.prompts.read_text(encoding="utf-8"))["prompts"]]
    seeds = [int(seed) for seed in args.seeds.split(",")]
    pool = load_pool(args.runs_root, prompts, seeds)
    baselines = [p for p in pool.values() if p["reference"] == "baseline_ddim20"]
    if not baselines:
        raise ValueError("pool has no 20-step DDIM baseline to anchor the quality floor")
    floor = baselines[0]["objectives"]["clip_score"] - args.max_clip_drop
    eligible = [p for p in pool.values() if p["objectives"]["clip_score"] >= floor]
    front = sorted(nondominated(eligible, BASE_OBJECTIVES), key=lambda p: p["objectives"]["latency_p50_ms"])
    selected = front
    if len(front) > args.max_finalists:
        best_clip = max(range(len(front)), key=lambda i: front[i]["objectives"]["clip_score"])
        picks = {0, best_clip}
        slots = args.max_finalists
        for k in range(slots):
            if len(picks) >= slots:
                break
            picks.add(round(k * (len(front) - 1) / max(1, slots - 1)))
        selected = [front[i] for i in sorted(picks)][: args.max_finalists]
    for point in selected:
        point["selection"] = "front"
    chosen = {p["run_id"] for p in selected}
    for point in pool.values():
        if point["reference"] and point["run_id"] not in chosen:
            selected.append({**point, "selection": "reference"})

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule": {"max_clip_drop": args.max_clip_drop, "clip_floor": floor, "max_finalists": args.max_finalists},
        "objectives": [{"name": name, "direction": direction} for name, direction in BASE_OBJECTIVES],
        "pool_unique_policies": len(pool),
        "eligible_count": len(eligible),
        "calibration_front_size": len(front),
        "points": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"pool {len(pool)} unique, eligible {len(eligible)}, front {len(front)}, locked {len(selected)} -> {args.output}")
    for point in selected:
        o = point["objectives"]
        print(f"  [{point['selection']}] {point['method']:<13} {o['latency_p50_ms']:7.1f} ms  CLIP {o['clip_score']:.3f}  {point['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
