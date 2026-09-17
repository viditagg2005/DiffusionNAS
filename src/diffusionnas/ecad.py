"""Evolutionary Caching to Accelerate Diffusion (ECAD) optimizer.

Adapts the ECAD genetic algorithm framework (Aggarwal et al., ICLR 2026)
to optimize post-hoc inference and caching policies for Stable Diffusion 1.5.
Discovers a Pareto-optimal frontier balancing image quality (CLIP score),
latency (p50 ms), and peak VRAM allocation without model retraining.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any

from .hypervolume import compute_hypervolume
from .pareto import BASE_OBJECTIVES, dominates, nondominated
from .quality import ClipScorer, DEFAULT_CLIP_MODEL
from .random_search import DEFAULT_REFERENCE_POINT, _simulate_objectives
from .runner import execute
from .search_space import (
    CHROMOSOME_LENGTH,
    decode_policy,
    encode_policy,
    policy_to_config,
    random_chromosome,
    read_objectives,
    validate_search_config,
)


class Individual:
    """An individual candidate in the ECAD population."""

    def __init__(
        self,
        chromosome: list[int],
        candidate_id: int,
        generation: int = 0,
    ) -> None:
        self.chromosome = list(chromosome)
        self.candidate_id = candidate_id
        self.generation = generation
        self.policy: dict[str, Any] = decode_policy(self.chromosome)
        self.run_dir: str | None = None
        self.objectives: dict[str, float] | None = None
        self.rank: int = 0
        self.crowding_distance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "chromosome": self.chromosome,
            "policy": self.policy,
            "run_dir": self.run_dir,
            "objectives": self.objectives,
            "rank": self.rank,
            "crowding_distance": round(self.crowding_distance, 4),
        }


def fast_nondominated_sort(individuals: list[Individual]) -> list[list[Individual]]:
    """Partition individuals into non-dominated Pareto fronts (NSGA-II)."""
    domination_counts = [0] * len(individuals)
    dominated_sets: list[list[int]] = [[] for _ in individuals]
    fronts: list[list[Individual]] = [[]]

    for p_idx, p in enumerate(individuals):
        assert p.objectives is not None
        for q_idx, q in enumerate(individuals):
            if p_idx == q_idx:
                continue
            assert q.objectives is not None
            if dominates(p.objectives, q.objectives, BASE_OBJECTIVES):
                dominated_sets[p_idx].append(q_idx)
            elif dominates(q.objectives, p.objectives, BASE_OBJECTIVES):
                domination_counts[p_idx] += 1

        if domination_counts[p_idx] == 0:
            p.rank = 0
            fronts[0].append(p)

    curr_rank = 0
    while curr_rank < len(fronts) and fronts[curr_rank]:
        next_front: list[Individual] = []
        for p in fronts[curr_rank]:
            p_idx = individuals.index(p)
            for q_idx in dominated_sets[p_idx]:
                domination_counts[q_idx] -= 1
                if domination_counts[q_idx] == 0:
                    q = individuals[q_idx]
                    q.rank = curr_rank + 1
                    next_front.append(q)
        curr_rank += 1
        if next_front:
            fronts.append(next_front)

    return fronts


def calculate_crowding_distance(front: list[Individual]) -> None:
    """Calculate crowding distance for individuals within a single front."""
    size = len(front)
    if size == 0:
        return
    for ind in front:
        ind.crowding_distance = 0.0

    if size <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return

    for obj_name, _ in BASE_OBJECTIVES:
        front.sort(key=lambda ind: ind.objectives[obj_name])  # type: ignore[index]
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        obj_min = front[0].objectives[obj_name]  # type: ignore[index]
        obj_max = front[-1].objectives[obj_name]  # type: ignore[index]
        obj_range = obj_max - obj_min

        if obj_range == 0:
            continue

        for i in range(1, size - 1):
            if front[i].crowding_distance != float("inf"):
                distance = (
                    front[i + 1].objectives[obj_name] - front[i - 1].objectives[obj_name]  # type: ignore[index]
                ) / obj_range
                front[i].crowding_distance += distance


def tournament_select(population: list[Individual], rng: random.Random) -> Individual:
    """Binary tournament selection using non-dominated rank and crowding distance."""
    if len(population) < 2:
        return population[0]
    i1, i2 = rng.sample(population, 2)
    if i1.rank < i2.rank:
        return i1
    if i2.rank < i1.rank:
        return i2
    if i1.crowding_distance > i2.crowding_distance:
        return i1
    return i2


def crossover(
    parent1: list[int],
    parent2: list[int],
    rng: random.Random,
    crossover_rate: float = 0.9,
) -> tuple[list[int], list[int]]:
    """Two-point crossover between binary chromosomes."""
    if rng.random() > crossover_rate:
        return list(parent1), list(parent2)

    length = len(parent1)
    pt1 = rng.randint(0, length - 2)
    pt2 = rng.randint(pt1 + 1, length - 1)

    c1 = list(parent1)
    c2 = list(parent2)
    c1[pt1:pt2] = parent2[pt1:pt2]
    c2[pt1:pt2] = parent1[pt1:pt2]
    return c1, c2


def mutate(
    chromosome: list[int],
    rng: random.Random,
    mutation_rate: float = 0.05,
) -> list[int]:
    """Bit-flip mutation."""
    mutated = list(chromosome)
    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]
    return mutated


def initial_seeds() -> list[list[int]]:
    """Seed initial population with known representative configurations."""
    seeds = [
        # Baseline reference: DDIM, 20 steps
        encode_policy({"method": "baseline", "scheduler": "ddim", "num_inference_steps": 20, "guidance_scale": 7.5}),
        # DeepCache heuristic: interval 3, branch 0
        encode_policy({
            "method": "deepcache",
            "scheduler": "ddim",
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "deepcache_interval": 3,
            "deepcache_branch_id": 0,
        }),
        # Aggressive DeepCache: interval 5, branch 2
        encode_policy({
            "method": "deepcache",
            "scheduler": "dpm_solver++",
            "num_inference_steps": 12,
            "guidance_scale": 7.5,
            "deepcache_interval": 5,
            "deepcache_branch_id": 2,
        }),
        # Fast AutoDiffusion schedule
        encode_policy({
            "method": "autodiffusion",
            "scheduler": "dpm_solver++",
            "num_inference_steps": 8,
            "guidance_scale": 7.5,
            "timestep_pattern": "trailing",
        }),
    ]
    return seeds


def run_ecad_search(
    config: dict[str, Any],
    project_root: Path,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
    clip_model_id: str = DEFAULT_CLIP_MODEL,
    score_device: str = "cuda",
    batch_size: int = 8,
    reference_point: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Execute ECAD evolutionary search over diffusion inference policies.

    Parameters
    ----------
    config : dict[str, Any]
        Search configuration adhering to validate_search_config().
    project_root : Path
        Root path of the repository.
    output_dir : str | Path | None
        Target directory for artifacts.
    dry_run : bool
        If True, run pipeline without GPU execution using simulated
        telemetry.
    clip_model_id : str
        Model ID for CLIP scoring.
    score_device : str
        Device for CLIP evaluation.
    batch_size : int
        Batch size for CLIP evaluation.
    reference_point : dict[str, float] | None
        Reference point for hypervolume calculation.

    Returns
    -------
    dict[str, Any]
        Complete search results including Pareto front and generation logs.
    """
    validate_search_config(config)

    budget = config.get("budget", 100)
    pop_size = config.get("population_size", 20)
    crossover_rate = float(config.get("crossover_rate", 0.9))
    mutation_rate = float(config.get("mutation_rate", 0.05))
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
        target_dir = project_root / config.get("output_root", "artifacts/search/ecad")
    else:
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = target_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ref_point = dict(reference_point or DEFAULT_REFERENCE_POINT)

    scorer = None
    if not dry_run:
        scorer = ClipScorer(model_id=clip_model_id, device=score_device, batch_size=batch_size)

    evaluated_archive: list[Individual] = []
    generation_logs: list[dict[str, Any]] = []
    hv_history: list[dict[str, Any]] = []
    start_time = time.time()

    def evaluate_individual(ind: Individual) -> None:
        candidate_config = policy_to_config(
            policy=ind.policy,
            prompts=prompts,
            seeds=seeds,
            runtime=runtime,
            output_root=str(runs_dir.relative_to(project_root) if runs_dir.is_relative_to(project_root) else runs_dir),
            save_images=True,
        )
        run_dir = execute(candidate_config, project_root, dry_run=dry_run)
        ind.run_dir = str(run_dir)

        if dry_run:
            ind.objectives = _simulate_objectives(ind.policy, ind.chromosome)
        else:
            assert scorer is not None
            scorer.score_run(run_dir)
            extracted = read_objectives(run_dir)
            if extracted is None:
                raise RuntimeError(f"failed to extract objectives from completed run: {run_dir}")
            ind.objectives = extracted

        evaluated_archive.append(ind)

        # Track hypervolume
        all_dicts = [cand.to_dict() for cand in evaluated_archive]
        current_front = nondominated(all_dicts, BASE_OBJECTIVES)
        hv = compute_hypervolume(
            [cand["objectives"] for cand in current_front],
            ref_point,
            BASE_OBJECTIVES,
        )
        hv_history.append({
            "evaluation": len(evaluated_archive),
            "hypervolume": round(hv, 4),
            "nondominated_count": len(current_front),
        })

    # 1. Initialize Generation 0
    population: list[Individual] = []
    initial_chromosomes = initial_seeds()
    for chrom in initial_chromosomes:
        if len(population) < pop_size and len(evaluated_archive) < budget:
            ind = Individual(chrom, candidate_id=len(evaluated_archive), generation=0)
            evaluate_individual(ind)
            population.append(ind)

    while len(population) < pop_size and len(evaluated_archive) < budget:
        chrom = random_chromosome(rng)
        ind = Individual(chrom, candidate_id=len(evaluated_archive), generation=0)
        evaluate_individual(ind)
        population.append(ind)

    fronts = fast_nondominated_sort(population)
    for f in fronts:
        calculate_crowding_distance(f)

    gen_idx = 0
    generation_logs.append({
        "generation": gen_idx,
        "evaluations_total": len(evaluated_archive),
        "front_0_size": len(fronts[0]) if fronts else 0,
        "population": [ind.to_dict() for ind in population],
    })

    # 2. Evolutionary Loop
    while len(evaluated_archive) < budget:
        gen_idx += 1
        offspring: list[Individual] = []

        # Create offspring
        while len(offspring) < pop_size and len(evaluated_archive) + len(offspring) < budget:
            parent1 = tournament_select(population, rng)
            parent2 = tournament_select(population, rng)
            c1_bits, c2_bits = crossover(parent1.chromosome, parent2.chromosome, rng, crossover_rate)
            c1_mutated = mutate(c1_bits, rng, mutation_rate)
            offspring.append(Individual(c1_mutated, candidate_id=len(evaluated_archive) + len(offspring), generation=gen_idx))
            if len(offspring) < pop_size and len(evaluated_archive) + len(offspring) < budget:
                c2_mutated = mutate(c2_bits, rng, mutation_rate)
                offspring.append(Individual(c2_mutated, candidate_id=len(evaluated_archive) + len(offspring), generation=gen_idx))

        # Evaluate offspring
        for ind in offspring:
            evaluate_individual(ind)

        # Environmental selection (mu + lambda)
        combined = population + offspring
        fronts = fast_nondominated_sort(combined)
        new_population: list[Individual] = []

        for front in fronts:
            calculate_crowding_distance(front)
            if len(new_population) + len(front) <= pop_size:
                new_population.extend(front)
            else:
                needed = pop_size - len(new_population)
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                new_population.extend(front[:needed])
                break

        population = new_population
        generation_logs.append({
            "generation": gen_idx,
            "evaluations_total": len(evaluated_archive),
            "front_0_size": len(fronts[0]) if fronts else 0,
            "population": [ind.to_dict() for ind in population],
        })

    elapsed_time = time.time() - start_time
    all_archive_dicts = [ind.to_dict() for ind in evaluated_archive]
    final_front = nondominated(all_archive_dicts, BASE_OBJECTIVES)
    sorted_front = sorted(final_front, key=lambda p: p["objectives"]["latency_p50_ms"])

    search_result: dict[str, Any] = {
        "schema_version": 1,
        "search_method": "ecad",
        "status": "dry_run" if dry_run else "succeeded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "budget": budget,
        "population_size": pop_size,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "seed": seed,
        "total_generations": gen_idx + 1,
        "evaluated_count": len(evaluated_archive),
        "reference_point": ref_point,
        "objectives": [{"name": name, "direction": direction} for name, direction in BASE_OBJECTIVES],
        "nondominated_count": len(sorted_front),
        "pareto_front": sorted_front,
        "hypervolume_history": hv_history,
        "candidates": all_archive_dicts,
        "search_cost": {
            "wall_clock_seconds": round(elapsed_time, 2),
            "total_evaluations": len(evaluated_archive),
        },
    }

    # Save search_results.json
    (target_dir / "search_results.json").write_text(
        json.dumps(search_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Save generations.json
    (target_dir / "generations.json").write_text(
        json.dumps({"generations": generation_logs}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Save pareto_front.json
    pareto_export = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "search_method": "ecad",
        "objectives": [{"name": name, "direction": direction} for name, direction in BASE_OBJECTIVES],
        "eligible_count": len(evaluated_archive),
        "nondominated_count": len(sorted_front),
        "points": sorted_front,
    }
    (target_dir / "pareto_front.json").write_text(
        json.dumps(pareto_export, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Save pareto_front.csv
    csv_path = target_dir / "pareto_front.csv"
    fieldnames = ["candidate_id", "generation", "method", "scheduler", "num_inference_steps", "guidance_scale"] + [
        name for name, _ in BASE_OBJECTIVES
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for point in sorted_front:
            policy = point["policy"]
            writer.writerow({
                "candidate_id": point["candidate_id"],
                "generation": point["generation"],
                "method": policy["method"],
                "scheduler": policy["scheduler"],
                "num_inference_steps": policy["num_inference_steps"],
                "guidance_scale": policy["guidance_scale"],
                **point["objectives"],
            })

    return search_result
