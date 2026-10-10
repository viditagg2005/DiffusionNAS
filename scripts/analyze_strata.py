#!/usr/bin/env python3
"""Per-stratum quality, fronts and paired baseline differences for scored runs.

Uses the caption-complexity strata stored in the prompt file. CPU only; reads
existing measurements and CLIP scores and never re-runs generation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas.holdout import spearman  # noqa: E402
from diffusionnas.pareto import BASE_OBJECTIVES, nondominated  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def policy_key(config: dict[str, Any]) -> str:
    return json.dumps(
        {
            "method": config["method"],
            "inference": config["inference"],
            "deepcache": config.get("deepcache"),
            "flexiffusion": config.get("flexiffusion"),
        },
        sort_keys=True,
    )


def policy_label(config: dict[str, Any]) -> str:
    inference = config["inference"]
    label = f"{config['method']}/{inference['scheduler']}/{inference['num_inference_steps']}st/g{inference['guidance_scale']}"
    if config.get("deepcache"):
        label += f"/i{config['deepcache']['interval']}b{config['deepcache']['branch_id']}"
    if inference.get("timestep_pattern"):
        label += f"/{inference['timestep_pattern']}"
    return label


def load_run(run_dir: Path, strata: list[str]) -> dict[str, Any] | None:
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        quality = read_jsonl(run_dir / "quality_measurements.jsonl")
        timing = read_jsonl(run_dir / "measurements.jsonl")
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("status") != "succeeded" or len(config["prompts"]) != len(strata):
        return None
    clip_by_prompt: dict[int, list[float]] = defaultdict(list)
    for row in quality:
        clip_by_prompt[row["prompt_index"]].append(row["clip_score"])
    latency_by_stratum: dict[str, list[float]] = defaultdict(list)
    for row in timing:
        latency_by_stratum[strata[row["prompt_index"]]].append(row["latency_ms"])
    return {
        "run_id": manifest["run_id"],
        "key": policy_key(config),
        "label": policy_label(config),
        "clip_by_prompt": {index: statistics.fmean(values) for index, values in clip_by_prompt.items()},
        "latency_by_stratum": latency_by_stratum,
        "peak_allocated_mb": max(row["peak_allocated_mb"] for row in timing),
    }


def paired_bootstrap(diffs: list[float], draws: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(draws))
    return [means[int(0.025 * draws)], means[int(0.975 * draws) - 1]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True, help="prompt file with per-prompt strata")
    parser.add_argument("--runs-root", type=Path, action="append", required=True)
    parser.add_argument("--baseline-run", type=Path, help="run used for paired per-stratum differences")
    parser.add_argument("--latency-budget-ms", type=float, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()

    strata = [item["stratum"] for item in json.loads(args.prompts.read_text(encoding="utf-8"))["prompts"]]
    names = sorted(set(strata))
    runs = []
    for root in args.runs_root:
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            run = load_run(run_dir, strata)
            if run is not None:
                runs.append(run)
    # Duplicate policies give identical images (fixed seeds); keep the first measured run.
    unique: dict[str, dict[str, Any]] = {}
    duplicates: dict[str, int] = defaultdict(int)
    for run in runs:
        duplicates[run["key"]] += 1
        unique.setdefault(run["key"], run)
    policies = list(unique.values())

    baseline = load_run(args.baseline_run, strata) if args.baseline_run else None
    report: dict[str, Any] = {
        "prompt_file": str(args.prompts),
        "runs_read": len(runs),
        "unique_policies": len(policies),
        "strata": {},
    }
    for name in names:
        indices = [index for index, label in enumerate(strata) if label == name]
        points = []
        for policy in policies:
            points.append(
                {
                    "run_id": policy["run_id"],
                    "label": policy["label"],
                    "duplicates": duplicates[policy["key"]],
                    "objectives": {
                        "clip_score": statistics.fmean(policy["clip_by_prompt"][i] for i in indices),
                        "latency_p50_ms": statistics.median(policy["latency_by_stratum"][name]),
                        "peak_allocated_mb": policy["peak_allocated_mb"],
                    },
                }
            )
        front = nondominated(points, BASE_OBJECTIVES)
        entry: dict[str, Any] = {
            "prompt_count": len(indices),
            "front": sorted(
                ({"run_id": p["run_id"], "label": p["label"], **p["objectives"]} for p in front),
                key=lambda p: p["latency_p50_ms"],
            ),
            "best_under_budget": {},
        }
        for budget in args.latency_budget_ms:
            eligible = [p for p in points if p["objectives"]["latency_p50_ms"] <= budget]
            if eligible:
                best = max(eligible, key=lambda p: p["objectives"]["clip_score"])
                entry["best_under_budget"][str(budget)] = {"label": best["label"], **best["objectives"]}
        if baseline is not None:
            entry["paired_vs_baseline"] = {}
            for policy in policies:
                diffs = [policy["clip_by_prompt"][i] - baseline["clip_by_prompt"][i] for i in indices]
                entry["paired_vs_baseline"][policy["label"]] = {
                    "mean_clip_difference": statistics.fmean(diffs),
                    "ci95": paired_bootstrap(diffs, args.bootstrap_draws, seed=len(indices)),
                }
        report["strata"][name] = entry

    # Does the CLIP ordering of policies change between strata?
    report["clip_rank_correlation"] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            left_idx = [k for k, label in enumerate(strata) if label == left]
            right_idx = [k for k, label in enumerate(strata) if label == right]
            report["clip_rank_correlation"][f"{left}~{right}"] = spearman(
                [statistics.fmean(p["clip_by_prompt"][k] for k in left_idx) for p in policies],
                [statistics.fmean(p["clip_by_prompt"][k] for k in right_idx) for p in policies],
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "strata"}, indent=2))
    for name, entry in report["strata"].items():
        print(f"\n[{name}] n={entry['prompt_count']} front={len(entry['front'])}")
        for budget, best in entry["best_under_budget"].items():
            print(f"  best <= {budget} ms: {best['label']} CLIP {best['clip_score']:.3f} @ {best['latency_p50_ms']:.0f} ms")
        for label, diff in entry.get("paired_vs_baseline", {}).items():
            low, high = diff["ci95"]
            print(f"  {label}: dCLIP {diff['mean_clip_difference']:+.3f} [{low:+.3f}, {high:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
