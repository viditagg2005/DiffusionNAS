#!/usr/bin/env python3
"""Lock v3 validation finalists from all calibration searches with a rule fixed before they ran.

1. Pool every candidate of every seed and optimizer (all must report one GPU model).
2. Latency is made comparable across jobs: each candidate's latency is divided by
   its job's reference latency and multiplied by the median reference latency.
3. Merge identical execution signatures (median normalized latency).
4. Eligible: CLIP >= reference CLIP - 1.0, fidelity >= 75, black images <= reference + 2.
5. Keep the 4-objective nondominated set (CLIP, fidelity, latency, U-Net MB).
6. Pick at most ``--max-finalists``: the best point on each objective, then greedy
   farthest-point selection in min-max normalized objective space.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas import nas_space  # noqa: E402
from diffusionnas.nas_search import SEARCH_OBJECTIVES  # noqa: E402
from diffusionnas.pareto import nondominated  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def local_run(path: str, runs_dir: Path) -> Path:
    """Map a run path recorded on the cluster onto the locally copied tree."""
    return runs_dir / Path(path).name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nas-root", type=Path, default=ROOT / "artifacts" / "nas")
    parser.add_argument("--max-finalists", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "nas" / "finalists.json")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; finalists are locked once. Delete it only with a logged reason.")

    references, pool, gpus = {}, {}, set()
    for seed_dir in sorted(args.nas_root.glob("seed*")):
        ref_run = next((seed_dir / "reference" / "runs").iterdir())
        ref_summary = read(ref_run / "summary.json")
        references[seed_dir.name] = {
            "latency": ref_summary["latency_ms"]["p50"],
            "clip": ref_summary["quality_metrics"]["clip_score"]["mean"],
            "black": ref_summary["quality_metrics"]["black_images"],
        }
        gpus.add(read(ref_run / "manifest.json")["provenance"]["gpu"]["name"])
        for method in ("random", "nsga2"):
            results = read(seed_dir / method / "search_results.json")
            for candidate in results["candidates"]:
                run = local_run(candidate["run_dir"], seed_dir / method / "runs")
                if (run / "manifest.json").exists():
                    gpus.add(read(run / "manifest.json")["provenance"]["gpu"]["name"])
                key = nas_space.signature(candidate["policy"])
                entry = pool.setdefault(key, {"policy": candidate["policy"], "objectives": dict(candidate["objectives"]),
                                              "relative_latency": [], "sources": []})
                entry["relative_latency"].append(candidate["objectives"]["latency_p50_ms"] / references[seed_dir.name]["latency"])
                entry["sources"].append(f"{seed_dir.name}/{method}/{candidate['candidate_id']}")
    if len(gpus) != 1:
        raise SystemExit(f"searches ran on {sorted(gpus)}; latency is not comparable")
    pooled_latency = statistics.median(r["latency"] for r in references.values())
    ref_clip = statistics.median(r["clip"] for r in references.values())
    ref_black = max(r["black"] for r in references.values())
    for entry in pool.values():
        entry["objectives"]["latency_p50_ms"] = statistics.median(entry["relative_latency"]) * pooled_latency

    eligible = [e for e in pool.values()
                if e["objectives"]["clip_score"] >= ref_clip - 1.0
                and e["objectives"]["clip_image_fidelity"] >= 75.0
                and e["objectives"]["black_images"] <= ref_black + 2]
    front = nondominated(eligible, SEARCH_OBJECTIVES)
    if not front:
        raise SystemExit("no eligible candidates")

    names = [name for name, _ in SEARCH_OBJECTIVES]
    lows = {n: min(p["objectives"][n] for p in front) for n in names}
    spans = {n: (max(p["objectives"][n] for p in front) - lows[n]) or 1.0 for n in names}

    def vector(point):
        return [(point["objectives"][n] - lows[n]) / spans[n] for n in names]

    chosen: list[dict] = []
    for name, direction in SEARCH_OBJECTIVES:
        best = (max if direction == "max" else min)(front, key=lambda p: p["objectives"][name])
        if best not in chosen:
            chosen.append(best)
    while len(chosen) < min(args.max_finalists, len(front)):
        remaining = [p for p in front if p not in chosen]
        far = max(remaining, key=lambda p: min(sum((a - b) ** 2 for a, b in zip(vector(p), vector(c))) for c in chosen))
        chosen.append(far)
    chosen = chosen[: args.max_finalists]

    result = {
        "rule": "pool all seeds/optimizers; normalize latency by job reference; merge signatures; CLIP >= ref-1.0, "
                "fidelity >= 75, black <= ref+2; 4-objective front; per-objective best then farthest-point",
        "gpu": gpus.pop(),
        "pooled_reference_latency_ms": pooled_latency,
        "reference_clip": ref_clip,
        "pool_unique_policies": len(pool),
        "eligible_count": len(eligible),
        "front_size": len(front),
        "finalists": [{"name": f"nas_{i:02d}", **p} for i, p in enumerate(sorted(chosen, key=lambda p: p["objectives"]["latency_p50_ms"]))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"pool {len(pool)}, eligible {len(eligible)}, front {len(front)}, locked {len(chosen)} -> {args.output}")
    for f in result["finalists"]:
        o, a = f["objectives"], f["policy"]["architecture"]
        print(f"  {f['name']} {f['policy']['method']:12s} {f['policy']['num_inference_steps']:2d}st {a['weight_precision']} "
              f"-{len(a['removed_units'])}u  {o['latency_p50_ms']:6.1f} ms  CLIP {o['clip_score']:.2f}  "
              f"fid {o['clip_image_fidelity']:.1f}  {o['unet_weight_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
