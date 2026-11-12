#!/usr/bin/env python3
"""Export DiffNAS-Bench (data, metadata, surrogates) from all real v3 evaluations.

Sources: artifacts/nas (seeds 1-3, random + NSGA-II) and, when present, artifacts/nas_c2
(all arms, seeds 1-6). Latency is normalized by each job's own reference and rescaled to the
median reference. Output: artifacts/benchmark/ (git-ignored until release).
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diffusionnas import nas_space  # noqa: E402
from diffusionnas.bench import DIRECTIONS, OBJECTIVE_SETS  # noqa: E402
from diffusionnas.surrogate import Surrogate, cross_validate  # noqa: E402


def collect() -> list[dict]:
    records, references = [], {}
    for root in (ROOT / "artifacts/nas", ROOT / "artifacts/nas_c2"):
        for seed_dir in sorted(root.glob("seed*")):
            ref_runs = list((seed_dir / "reference" / "runs").glob("*"))
            if not ref_runs:
                continue
            reference = json.loads((ref_runs[0] / "summary.json").read_text())
            gpu = json.loads((ref_runs[0] / "manifest.json").read_text())["provenance"]["gpu"]["name"]
            references[(root.name, seed_dir.name)] = reference["latency_ms"]["p50"]
            for results_path in sorted(seed_dir.glob("*/search_results.json")):
                results = json.loads(results_path.read_text())
                if results.get("status") != "succeeded":
                    continue
                for candidate in results["candidates"]:
                    records.append({"policy": candidate["policy"], "objectives": dict(candidate["objectives"]),
                                    "source": f"{root.name}/{seed_dir.name}/{results_path.parent.name}",
                                    "job": (root.name, seed_dir.name), "gpu": gpu})
    pooled = statistics.median(references.values())
    unique: dict[str, dict] = {}
    for record in records:
        record["objectives"]["latency_p50_ms_raw"] = record["objectives"]["latency_p50_ms"]
        record["objectives"]["latency_p50_ms"] *= pooled / references[record["job"]]
        record.pop("job")
        key = nas_space.signature(record["policy"])
        if key in unique:  # keep one record per configuration; median latency across repeats
            kept = unique[key]
            kept.setdefault("latency_repeats", [kept["objectives"]["latency_p50_ms"]]).append(record["objectives"]["latency_p50_ms"])
            kept["objectives"]["latency_p50_ms"] = statistics.median(kept["latency_repeats"])
        else:
            unique[key] = record
    return list(unique.values()), pooled


def main() -> int:
    out = ROOT / "artifacts/benchmark"
    out.mkdir(parents=True, exist_ok=True)
    records, pooled = collect()
    gpus = sorted({r["gpu"] for r in records})
    rows = [{"policy": r["policy"], "objectives": r["objectives"], "group": r["source"].split("/")[1]} for r in records]
    reference_point = {"clip_score": 25.2, "clip_image_fidelity": 75.0, "latency_p50_ms": 600.0,
                       "gflops_per_image": 36000.0, "peak_allocated_mb": 3500.0, "unet_weight_mb": 1700.0}
    best = {n: (max if DIRECTIONS[n] == "max" else min)(r["objectives"][n] for r in records) for n in DIRECTIONS}
    surrogates, accuracy = {}, {}
    for key, names in OBJECTIVE_SETS.items():
        objectives = [(n, DIRECTIONS[n]) for n in names]
        surrogates[key] = Surrogate(objectives, seed=0, family="extratrees").fit(rows)
        accuracy[key] = cross_validate(rows, objectives, family="extratrees")
    metadata = {
        "name": "DiffNAS-Bench", "version": "0.1",
        "model": "stable-diffusion-v1-5/stable-diffusion-v1-5, 512x512, batch 1, fp16 activations",
        "gpus": gpus, "records": len(records), "pooled_reference_latency_ms": pooled,
        "objectives": {n: DIRECTIONS[n] for n in DIRECTIONS}, "objective_sets": OBJECTIVE_SETS,
        "reference_point": reference_point, "scaling_low": best,
        "search_space": {"units": list(nas_space.UNITS), "genome_length": nas_space.GENOME_LENGTH},
        "calibration": "48 COCO 2014 captions (data/prompts/calibration.json) x seeds 42, 123",
        "surrogate": "ExtraTrees (200 trees, min leaf 2) per objective on policy features (diffusionnas.surrogate.features)",
        "surrogate_accuracy": accuracy,
    }
    (out / "data.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "surrogates.pkl").write_bytes(pickle.dumps(surrogates))
    print(f"{len(records)} unique configurations from {gpus} -> {out}")
    for key, block in accuracy.items():
        print(key, {n: round(v["leave_seed_out_spearman"], 3) for n, v in block.items() if n != "scored_points"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
