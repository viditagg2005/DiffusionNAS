"""Random search optimizer for diffusion inference policies.

Serves as the mandatory control baseline in the benchmarking protocol.
Uniformly samples candidates from the shared policy search space,
evaluates them under the locked budget, tracks the running hypervolume
progression, and exports the final Pareto frontier.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any

from .hypervolume import compute_hypervolume, default_reference_point
from .pareto import BASE_OBJECTIVES, nondominated
from .quality import ClipScorer, DEFAULT_CLIP_MODEL
from .runner import execute
from .search_space import (
    CHROMOSOME_LENGTH,
    decode_policy,
    policy_to_config,
    random_chromosome,
    read_objectives,
    validate_search_config,
)


DEFAULT_REFERENCE_POINT: dict[str, float] = {
    "clip_score": 0.0,
    "latency_p50_ms": 5000.0,
    "peak_allocated_mb": 16000.0,
}


def _simulate_objectives(policy: dict[str, Any], chromosome: list[int]) -> dict[str, float]:
    """Provide deterministic simulated objectives for dry-run validation."""
    nfe = policy.get("num_inference_steps", 20)
    method = policy.get("method", "baseline")

    # Base latency proportional to NFE
    base_latency = 15.0 * nfe
    if method == "deepcache":
        interval = policy.get("deepcache_interval", 3)
        # DeepCache skips full UNet passes, reducing latency
        speedup = 1.0 - (1.0 / (interval + 1.0)) * 0.4
        latency = base_latency * speedup
        peak_vram = 2200.0 + (12 - policy.get("deepcache_branch_id", 0)) * 15.0
    elif method == "autodiffusion":
        latency = base_latency * 0.95
        peak_vram = 2050.0
    else:  # baseline
        latency = base_latency
        peak_vram = 2000.0

    # Quality: higher steps generally improve quality slightly, slight noise from chromosome bits
    bit_factor = sum(chromosome) / max(1, len(chromosome))
    clip = 22.0 + 4.0 * (nfe / 20.0) + 2.0 * bit_factor
    if method == "deepcache" and policy.get("deepcache_interval", 3) > 4:
        clip -= 1.5  # aggressive caching slightly reduces quality

    return {
        "clip_score": round(clip, 4),
        "latency_p50_ms": round(latency, 2),
        "peak_allocated_mb": round(peak_vram, 1),
    }


def run_random_search(
    config: dict[str, Any],
    project_root: Path,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    clip_model_id: str = DEFAULT_CLIP_MODEL,
    score_device: str = "cuda",
    batch_size: int = 8,
    reference_point: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Execute random search over the diffusion policy space.

    Parameters
    ----------
    config : dict[str, Any]
        Search configuration adhering to validate_search_config().
    project_root : Path
        Root path of the repository.
    output_dir : str | Path | None
        Directory where search artifacts are saved. Defaults to
        config['output_root'].
    dry_run : bool
        If True, run pipeline without GPU execution using simulated
        telemetry.
    clip_model_id : str
        HuggingFace model ID for CLIP scoring.
    score_device : str
        Device for CLIP scoring ('cuda' or 'cpu').
    batch_size : int
        Batch size for CLIP scoring.
    reference_point : dict[str, float] | None
        Reference point for hypervolume calculation.

    Returns
    -------
    dict[str, Any]
        Complete search results including candidate archive, final
        Pareto front, and hypervolume history.
    """
    validate_search_config(config)

    budget = config.get("budget", 100)
    seed = config.get("seed", 42)
    prompts = config["prompts"]
    seeds = config["seeds"]
    runtime = config.get("runtime", {
        "device": "cuda",
        "precision": "float16",
        "warmup_runs": 1,
        "repetitions": 3,
    })

    rng = random.Random(seed)

    if output_dir is None:
        target_dir = project_root / config.get("output_root", "artifacts/search/random")
    else:
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = target_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ref_point = dict(reference_point or DEFAULT_REFERENCE_POINT)

    scorer = None
    if not dry_run:
        scorer = ClipScorer(model_id=clip_model_id, device=score_device, batch_size=batch_size)

    candidates: list[dict[str, Any]] = []
    hv_history: list[dict[str, Any]] = []
    start_time = time.time()

    for idx in range(budget):
        chromosome = random_chromosome(rng)
        policy = decode_policy(chromosome)

        candidate_config = policy_to_config(
            policy=policy,
            prompts=prompts,
            seeds=seeds,
            runtime=runtime,
            output_root=str(runs_dir.relative_to(project_root) if runs_dir.is_relative_to(project_root) else runs_dir),
            save_images=True,
        )

        run_dir = execute(candidate_config, project_root, dry_run=dry_run)

        if dry_run:
            objectives = _simulate_objectives(policy, chromosome)
        else:
            assert scorer is not None
            scorer.score_run(run_dir)
            extracted = read_objectives(run_dir)
            if extracted is None:
                raise RuntimeError(f"failed to extract objectives from completed run: {run_dir}")
            objectives = extracted

        candidate_record = {
            "candidate_id": idx,
            "chromosome": chromosome,
            "policy": policy,
            "run_dir": str(run_dir),
            "objectives": objectives,
        }
        candidates.append(candidate_record)

        # Calculate running non-dominated front and hypervolume
        current_valid = [c for c in candidates if c.get("objectives")]
        current_front = nondominated(current_valid, BASE_OBJECTIVES)
        hv = compute_hypervolume(
            [c["objectives"] for c in current_front],
            ref_point,
            BASE_OBJECTIVES,
        )
        hv_history.append({
            "evaluation": idx + 1,
            "hypervolume": round(hv, 4),
            "nondominated_count": len(current_front),
        })

    elapsed_time = time.time() - start_time
    final_front = nondominated(candidates, BASE_OBJECTIVES)
    sorted_front = sorted(final_front, key=lambda p: p["objectives"]["latency_p50_ms"])

    search_result: dict[str, Any] = {
        "schema_version": 1,
        "search_method": "random",
        "status": "dry_run" if dry_run else "succeeded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "budget": budget,
        "seed": seed,
        "evaluated_count": len(candidates),
        "reference_point": ref_point,
        "objectives": [{"name": name, "direction": direction} for name, direction in BASE_OBJECTIVES],
        "nondominated_count": len(sorted_front),
        "pareto_front": sorted_front,
        "hypervolume_history": hv_history,
        "candidates": candidates,
        "search_cost": {
            "wall_clock_seconds": round(elapsed_time, 2),
            "total_evaluations": len(candidates),
        },
    }

    # Write search_results.json
    (target_dir / "search_results.json").write_text(
        json.dumps(search_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Write pareto_front.json matching harness schema
    pareto_export = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "search_method": "random",
        "objectives": [{"name": name, "direction": direction} for name, direction in BASE_OBJECTIVES],
        "eligible_count": len(candidates),
        "nondominated_count": len(sorted_front),
        "points": sorted_front,
    }
    front_json_path = target_dir / "pareto_front.json"
    front_json_path.write_text(
        json.dumps(pareto_export, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Write pareto_front.csv
    csv_path = target_dir / "pareto_front.csv"
    fieldnames = ["candidate_id", "method", "scheduler", "num_inference_steps", "guidance_scale"] + [
        name for name, _ in BASE_OBJECTIVES
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for point in sorted_front:
            policy = point["policy"]
            writer.writerow({
                "candidate_id": point["candidate_id"],
                "method": policy["method"],
                "scheduler": policy["scheduler"],
                "num_inference_steps": policy["num_inference_steps"],
                "guidance_scale": policy["guidance_scale"],
                **point["objectives"],
            })

    return search_result

