"""v3 architecture + policy search: random search and NSGA-II on identical terms.

Both optimizers draw from ``nas_space``, never re-evaluate an execution
signature, evaluate exactly ``budget`` unique candidates, and score each
candidate on four search objectives (CLIP prompt alignment, CLIP image
fidelity to a reference run from the same GPU allocation, p50 latency, U-Net
weight MB). GFLOPs per image and peak VRAM are measured for every candidate and
reported, but kept out of selection: with 6 objectives almost every NSGA-II
candidate is nondominated and selection degenerates to crowding. NSGA-II starts
from random novel genomes (no hand-picked seeds), so any hypervolume gap is due
to the optimizer, not to a head start.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Callable

from . import nas_space
from .ecad import Individual, calculate_crowding_distance, fast_nondominated_sort, tournament_select
from .hypervolume import compute_hypervolume
from .pareto import nondominated
from .runner import execute
from .search_checkpoint import SearchCheckpoint
from .search_space import policy_to_config

SEARCH_OBJECTIVES = (
    ("clip_score", "max"),
    ("clip_image_fidelity", "max"),
    ("latency_p50_ms", "min"),
    ("unet_weight_mb", "min"),
)
REPORTED_OBJECTIVES = SEARCH_OBJECTIVES + (("gflops_per_image", "min"), ("peak_allocated_mb", "min"))


def read_nas_objectives(run_dir: Path) -> dict[str, float]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    quality = summary["quality_metrics"]
    return {
        "clip_score": float(quality["clip_score"]["mean"]),
        "clip_image_fidelity": float(quality["clip_image_fidelity"]["mean"]),
        "latency_p50_ms": float(summary["latency_ms"]["p50"]),
        "unet_weight_mb": summary["architecture"]["unet_weight_bytes"] / 2**20,
        "gflops_per_image": float(summary.get("gflops_per_image") or 0.0),
        "peak_allocated_mb": float(summary["peak_allocated_mb"]),
        "black_images": float(quality.get("black_images", 0)),
    }


def simulate_objectives(policy: dict[str, Any]) -> dict[str, float]:
    """Deterministic stand-in for dry runs; shape only, not a model of real behaviour."""
    removed = len(policy["architecture"]["removed_units"])
    int8 = policy["architecture"]["weight_precision"] == "int8"
    steps = policy["num_inference_steps"]
    cached = policy["method"] != "baseline"
    latency = 25.0 * steps * (0.6 if cached else 1.0) * (1.0 - 0.01 * removed) * (1.24 if int8 else 1.0)
    clip = 26.2 - 0.25 * removed - (0.5 if steps < 8 else 0.0) - (0.3 if cached else 0.0) + 0.01 * policy["guidance_scale"]
    fidelity = 100.0 - 3.0 * removed - (4.0 if int8 else 0.0) - (25.0 - steps) * 0.6 - (5.0 if cached else 0.0)
    weight = 1639.4 - 40.0 * removed
    return {
        "clip_score": round(clip, 4), "clip_image_fidelity": round(fidelity, 3), "latency_p50_ms": round(latency, 2),
        "unet_weight_mb": round(weight / (2.0 if int8 else 1.0), 2), "gflops_per_image": 1700.0 * steps,
        "peak_allocated_mb": round(1600.0 + weight / (2.0 if int8 else 1.0), 1), "black_images": 0.0,
    }


class Evaluator:
    """Runs one candidate end to end (generate, CLIP, fidelity) through the shared checkpoint."""

    def __init__(self, config: dict[str, Any], project_root: Path, target_dir: Path, checkpoint: SearchCheckpoint,
                 reference_run: Path | None, dry_run: bool, scorer: Any) -> None:
        self.config, self.project_root, self.checkpoint = config, project_root, checkpoint
        self.reference_run, self.dry_run, self.scorer = reference_run, dry_run, scorer
        runs_dir = target_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.output_root = str(runs_dir.relative_to(project_root) if runs_dir.is_relative_to(project_root) else runs_dir)
        self.archive: list[dict[str, Any]] = []
        self.hv_history: list[dict[str, Any]] = []
        self.reference_point = config["reference_point"]

    def __call__(self, candidate_id: int, genome: list[int], generation: int) -> dict[str, Any]:
        policy = nas_space.decode(genome)
        run_config = policy_to_config(policy, self.config["prompts"], self.config["seeds"], dict(self.config["runtime"]),
                                      self.output_root, save_images=True)

        def evaluate_new() -> tuple[Path, dict[str, float]]:
            run_dir = execute(run_config, self.project_root, dry_run=self.dry_run)
            if self.dry_run:
                return run_dir, simulate_objectives(policy)
            self.scorer.score_run(run_dir)
            self.scorer.fidelity_run(run_dir, self.reference_run)
            return run_dir, read_nas_objectives(run_dir)

        run_dir, objectives = self.checkpoint.evaluate(candidate_id, genome, generation, evaluate_new)
        record = {"candidate_id": candidate_id, "generation": generation, "chromosome": list(genome),
                  "policy": policy, "run_dir": str(run_dir), "objectives": objectives}
        self.archive.append(record)
        front = nondominated(self.archive, SEARCH_OBJECTIVES)
        self.hv_history.append({
            "evaluation": len(self.archive),
            "hypervolume": compute_hypervolume([p["objectives"] for p in front], self.reference_point, SEARCH_OBJECTIVES),
            "nondominated_count": len(front),
        })
        print(f"{self.config['search_method']} v3: {len(self.archive)}/{self.config['budget']} "
              f"({policy['method']}, -{len(policy['architecture']['removed_units'])} units, "
              f"{policy['architecture']['weight_precision']})", flush=True)
        return record


def _random(config: dict[str, Any], rng: random.Random, evaluate: Callable, seen: set[str]) -> None:
    for candidate_id in range(config["budget"]):
        genome = nas_space.novel(lambda: nas_space.random_genome(rng), seen, rng)
        seen.add(nas_space.signature(nas_space.decode(genome)))
        evaluate(candidate_id, genome, 0)


def _nsga2(config: dict[str, Any], rng: random.Random, evaluate: Callable, seen: set[str], log: list) -> None:
    budget, pop_size = config["budget"], config["population_size"]
    crossover_rate, mutation_rate = config["crossover_rate"], config["mutation_rate"]
    switch = config.get("method_switch_rate", 0.1)
    count = 0

    def make(genome: list[int], generation: int) -> Individual:
        nonlocal count
        seen.add(nas_space.signature(nas_space.decode(genome)))
        ind = Individual(genome, candidate_id=count, generation=generation, decoder=nas_space.decode)
        ind.objectives = evaluate(count, genome, generation)["objectives"]
        count += 1
        return ind

    population = [make(nas_space.novel(lambda: nas_space.random_genome(rng), seen, rng), 0)
                  for _ in range(min(pop_size, budget))]
    for front in fast_nondominated_sort(population, SEARCH_OBJECTIVES):
        calculate_crowding_distance(front, SEARCH_OBJECTIVES)
    generation = 0
    log.append({"generation": 0, "population": [ind.candidate_id for ind in population]})

    def breed() -> list[int]:
        parent1, parent2 = tournament_select(population, rng), tournament_select(population, rng)
        child = parent1.chromosome
        if rng.random() < crossover_rate:
            child, _ = nas_space.crossover(parent1.chromosome, parent2.chromosome, rng)
        if rng.random() < mutation_rate:
            child = nas_space.mutate(child, rng, switch)
        return child

    while count < budget:
        generation += 1
        offspring = []
        while len(offspring) < pop_size and count < budget:
            offspring.append(make(nas_space.novel(breed, seen, rng), generation))
        combined = population + offspring
        survivors: list[Individual] = []
        for front in fast_nondominated_sort(combined, SEARCH_OBJECTIVES):
            calculate_crowding_distance(front, SEARCH_OBJECTIVES)
            if len(survivors) + len(front) <= pop_size:
                survivors.extend(front)
            else:
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                survivors.extend(front[: pop_size - len(survivors)])
                break
        population = survivors
        log.append({"generation": generation, "population": [ind.candidate_id for ind in population]})


def validate_nas_config(config: dict[str, Any]) -> None:
    errors = []
    if config.get("search_method") not in {"random", "nsga2"}:
        errors.append("search_method must be 'random' or 'nsga2'")
    if config.get("search_space") != "v3":
        errors.append("search_space must be 'v3'")
    if not isinstance(config.get("budget"), int) or config["budget"] < 2:
        errors.append("budget must be an integer >= 2")
    if set(config.get("reference_point", {})) != {name for name, _ in SEARCH_OBJECTIVES}:
        errors.append(f"reference_point must give {[name for name, _ in SEARCH_OBJECTIVES]}")
    if config.get("search_method") == "nsga2":
        for key in ("population_size", "crossover_rate", "mutation_rate"):
            if key not in config:
                errors.append(f"nsga2 requires {key}")
    if errors:
        raise ValueError("; ".join(errors))


def run_nas_search(config: dict[str, Any], project_root: Path, output_dir: Path, reference_run: Path | None,
                   dry_run: bool = False, clip_model_id: str = "openai/clip-vit-large-patch14", score_device: str = "cuda",
                   batch_size: int = 16, max_walltime_seconds: float | None = None,
                   max_new_evaluations: int | None = None) -> dict[str, Any]:
    validate_nas_config(config)
    if not dry_run and reference_run is None:
        raise ValueError("a real search needs a reference run for fidelity")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = SearchCheckpoint(
        output_dir=output_dir, config={**config, "units": list(nas_space.UNITS),
                                       "reference_run": str(reference_run.resolve()) if reference_run else None},
        method=f"{config['search_method']}_v3", dry_run=dry_run, clip_model_id=clip_model_id,
        score_device=score_device, batch_size=batch_size, reference_point=config["reference_point"],
        max_walltime_seconds=max_walltime_seconds, max_new_evaluations=max_new_evaluations,
    )
    scorer = None
    if not dry_run:
        from .quality import ClipScorer

        scorer = ClipScorer(model_id=clip_model_id, device=score_device, batch_size=batch_size)
    evaluate = Evaluator(config, project_root, output_dir, checkpoint, reference_run, dry_run, scorer)
    rng = random.Random(config["seed"])
    seen: set[str] = set()
    generations: list[dict[str, Any]] = []
    if config["search_method"] == "random":
        _random(config, rng, evaluate, seen)
    else:
        _nsga2(config, rng, evaluate, seen, generations)

    front = sorted(nondominated(evaluate.archive, SEARCH_OBJECTIVES), key=lambda p: p["objectives"]["latency_p50_ms"])
    result = {
        "schema_version": 1,
        "search_method": config["search_method"],
        "search_space": "v3",
        "status": "dry_run" if dry_run else "succeeded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": config["seed"],
        "budget": config["budget"],
        "units": list(nas_space.UNITS),
        "reference_run": str(reference_run) if reference_run else None,
        "reference_point": config["reference_point"],
        "search_objectives": [{"name": n, "direction": d} for n, d in SEARCH_OBJECTIVES],
        "reported_objectives": [{"name": n, "direction": d} for n, d in REPORTED_OBJECTIVES],
        "evaluated_count": len(evaluate.archive),
        "final_hypervolume": evaluate.hv_history[-1]["hypervolume"] if evaluate.hv_history else 0.0,
        "hypervolume_history": evaluate.hv_history,
        "pareto_front": front,
        "candidates": evaluate.archive,
        "generations": generations,
        "wall_clock_seconds": round(checkpoint.elapsed_wall_seconds, 1),
    }
    (output_dir / "search_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint.mark_complete()
    return result
