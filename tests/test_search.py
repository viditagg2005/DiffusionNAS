"""CPU-only unit tests for Search Space, Hypervolume, Random Search, and ECAD."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from diffusionnas.ecad import (
    Individual,
    calculate_crowding_distance,
    crossover,
    fast_nondominated_sort,
    mutate,
    run_ecad_search,
)
from diffusionnas.hypervolume import (
    compute_hypervolume,
    default_reference_point,
)
from diffusionnas.random_search import run_random_search
from diffusionnas.schema import validate_config
from diffusionnas.search_cli import main as search_cli_main
from diffusionnas.search_space import (
    CHROMOSOME_LENGTH,
    decode_policy,
    encode_policy,
    generate_timesteps,
    policy_to_config,
    random_chromosome,
    sample_random_policy,
    validate_search_config,
)


class SearchSpaceTests(unittest.TestCase):
    def test_chromosome_length(self):
        rng = random.Random(42)
        chrom = random_chromosome(rng)
        self.assertEqual(len(chrom), CHROMOSOME_LENGTH)
        self.assertTrue(all(b in (0, 1) for b in chrom))

    def test_encode_decode_roundtrip(self):
        policy = {
            "method": "deepcache",
            "scheduler": "dpm_solver++",
            "num_inference_steps": 12,
            "guidance_scale": 7.5,
            "deepcache_interval": 3,
            "deepcache_branch_id": 2,
            "timestep_pattern": "linear",
        }
        chrom = encode_policy(policy)
        decoded = decode_policy(chrom)
        self.assertEqual(decoded["method"], "deepcache")
        self.assertEqual(decoded["scheduler"], "dpm_solver++")
        self.assertEqual(decoded["num_inference_steps"], 12)
        self.assertEqual(decoded["guidance_scale"], 7.5)
        self.assertEqual(decoded["deepcache_interval"], 3)
        self.assertEqual(decoded["deepcache_branch_id"], 2)

    def test_sample_random_policy_produces_valid_config(self):
        rng = random.Random(123)
        prompts = ["a scenic landscape"]
        seeds = [42]
        for _ in range(30):
            policy = sample_random_policy(rng)
            config = policy_to_config(policy, prompts, seeds)
            # Must pass standard harness validation without errors
            validate_config(config)

    def test_autodiffusion_timesteps_descend(self):
        for pattern in ("linear", "trailing", "leading"):
            for nfe in (4, 8, 12, 20):
                timesteps = generate_timesteps(nfe, pattern)
                self.assertEqual(len(timesteps), nfe)
                self.assertTrue(all(0 <= t <= 999 for t in timesteps))
                # Strictly descending
                self.assertTrue(all(a > b for a, b in zip(timesteps, timesteps[1:])))

    def test_validate_search_config(self):
        valid = {
            "search_method": "random",
            "budget": 10,
            "seed": 42,
            "prompts": ["photo"],
            "seeds": [1],
            "runtime": {"device": "cuda", "precision": "float16"},
        }
        validate_search_config(valid)

        invalid = dict(valid)
        invalid["budget"] = 0
        with self.assertRaises(ValueError):
            validate_search_config(invalid)


class HypervolumeTests(unittest.TestCase):
    def test_hypervolume_2d_known_value(self):
        # Two points in 2-D minimization:
        # P1 = (1, 3), P2 = (2, 2) with ref = (4, 4)
        # Dominance region:
        # P1 dominates [1, 4] x [3, 4] -> area = 3 x 1 = 3
        # P2 dominates [2, 4] x [2, 4] -> area = 2 x 2 = 4
        # Intersection: [2, 4] x [3, 4] -> area = 2 x 1 = 2
        # Union = 3 + 4 - 2 = 5.0
        points = [{"x": 1.0, "y": 3.0}, {"x": 2.0, "y": 2.0}]
        ref = {"x": 4.0, "y": 4.0}
        objectives = (("x", "min"), ("y", "min"))
        hv = compute_hypervolume(points, ref, objectives)
        self.assertAlmostEqual(hv, 5.0)

    def test_hypervolume_empty_set(self):
        ref = {"x": 10.0, "y": 10.0}
        objectives = (("x", "min"), ("y", "min"))
        self.assertEqual(compute_hypervolume([], ref, objectives), 0.0)

    def test_hypervolume_with_max_direction(self):
        # Maximization of quality: q1 = 8, q2 = 6; Min of latency: l1 = 5, l2 = 2
        # Ref: quality = 0, latency = 10
        # By symmetry with min: (-8, 5) vs (-6, 2) with ref (0, 10)
        points = [{"q": 8.0, "l": 5.0}, {"q": 6.0, "l": 2.0}]
        ref = {"q": 0.0, "l": 10.0}
        objectives = (("q", "max"), ("l", "min"))
        hv = compute_hypervolume(points, ref, objectives)
        self.assertGreater(hv, 0.0)

    def test_default_reference_point(self):
        points = [{"x": 2.0, "y": 5.0}, {"x": 4.0, "y": 3.0}]
        objectives = (("x", "min"), ("y", "max"))
        ref = default_reference_point(points, objectives, margin=0.1)
        self.assertGreater(ref["x"], 4.0)  # worse for min
        self.assertLess(ref["y"], 3.0)  # worse for max


class RandomSearchTests(unittest.TestCase):
    def test_random_search_dry_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "random_out"
            config = {
                "search_method": "random",
                "budget": 5,
                "seed": 42,
                "prompts": ["a photo of an astronaut"],
                "seeds": [7],
                "runtime": {
                    "device": "cuda",
                    "precision": "float16",
                    "warmup_runs": 0,
                    "repetitions": 1,
                },
                "output_root": str(output_path),
            }
            results = run_random_search(config, ROOT, output_dir=output_path, dry_run=True)

            self.assertEqual(results["status"], "dry_run")
            self.assertEqual(results["evaluated_count"], 5)
            self.assertEqual(len(results["hypervolume_history"]), 5)
            self.assertGreater(results["nondominated_count"], 0)

            self.assertTrue((output_path / "search_results.json").exists())
            self.assertTrue((output_path / "pareto_front.json").exists())
            self.assertTrue((output_path / "pareto_front.csv").exists())

            saved_results = json.loads((output_path / "search_results.json").read_text())
            self.assertEqual(saved_results["evaluated_count"], 5)


class ECADTests(unittest.TestCase):
    def test_crossover_and_mutation_invariants(self):
        rng = random.Random(99)
        p1 = random_chromosome(rng)
        p2 = random_chromosome(rng)
        c1, c2 = crossover(p1, p2, rng, crossover_rate=1.0)
        self.assertEqual(len(c1), CHROMOSOME_LENGTH)
        self.assertEqual(len(c2), CHROMOSOME_LENGTH)

        mutated = mutate(c1, rng, mutation_rate=0.5)
        self.assertEqual(len(mutated), CHROMOSOME_LENGTH)
        self.assertTrue(all(b in (0, 1) for b in mutated))

    def test_fast_nondominated_sort_and_crowding(self):
        # Create 3 individuals
        # ind0: (clip=30, lat=100, mem=2000) -> non-dominated
        # ind1: (clip=20, lat=200, mem=2500) -> dominated by ind0
        # ind2: (clip=31, lat=250, mem=2100) -> trade-off with ind0
        ind0 = Individual([0] * CHROMOSOME_LENGTH, candidate_id=0)
        ind0.objectives = {"clip_score": 30.0, "latency_p50_ms": 100.0, "peak_allocated_mb": 2000.0}

        ind1 = Individual([0] * CHROMOSOME_LENGTH, candidate_id=1)
        ind1.objectives = {"clip_score": 20.0, "latency_p50_ms": 200.0, "peak_allocated_mb": 2500.0}

        ind2 = Individual([0] * CHROMOSOME_LENGTH, candidate_id=2)
        ind2.objectives = {"clip_score": 31.0, "latency_p50_ms": 250.0, "peak_allocated_mb": 2100.0}

        fronts = fast_nondominated_sort([ind0, ind1, ind2])
        self.assertEqual(len(fronts), 2)
        self.assertEqual({ind.candidate_id for ind in fronts[0]}, {0, 2})
        self.assertEqual({ind.candidate_id for ind in fronts[1]}, {1})

        calculate_crowding_distance(fronts[0])
        # Two endpoints get infinity
        self.assertEqual(fronts[0][0].crowding_distance, float("inf"))
        self.assertEqual(fronts[0][1].crowding_distance, float("inf"))

    def test_ecad_dry_run_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "ecad_out"
            config = {
                "search_method": "ecad",
                "budget": 6,
                "population_size": 4,
                "crossover_rate": 0.9,
                "mutation_rate": 0.05,
                "seed": 42,
                "prompts": ["a photo of an astronaut"],
                "seeds": [7],
                "runtime": {
                    "device": "cuda",
                    "precision": "float16",
                    "warmup_runs": 0,
                    "repetitions": 1,
                },
                "output_root": str(output_path),
            }
            results = run_ecad_search(config, ROOT, output_dir=output_path, dry_run=True)

            self.assertEqual(results["status"], "dry_run")
            self.assertEqual(results["evaluated_count"], 6)
            self.assertGreater(results["nondominated_count"], 0)

            self.assertTrue((output_path / "search_results.json").exists())
            self.assertTrue((output_path / "generations.json").exists())
            self.assertTrue((output_path / "pareto_front.json").exists())
            self.assertTrue((output_path / "pareto_front.csv").exists())


class SearchCLITests(unittest.TestCase):
    def test_cli_random_dry_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg_path = Path(temp_dir) / "cfg.json"
            cfg_path.write_text(
                json.dumps({
                    "search_method": "random",
                    "budget": 2,
                    "seed": 1,
                    "prompts": ["test prompt"],
                    "seeds": [10],
                    "runtime": {"device": "cuda", "precision": "float16"},
                    "output_root": str(Path(temp_dir) / "out"),
                }),
                encoding="utf-8",
            )
            exit_code = search_cli_main(["random", "--config", str(cfg_path), "--dry-run"])
            self.assertEqual(exit_code, 0)

    def test_cli_ecad_dry_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg_path = Path(temp_dir) / "cfg.json"
            cfg_path.write_text(
                json.dumps({
                    "search_method": "ecad",
                    "budget": 3,
                    "population_size": 2,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.05,
                    "seed": 1,
                    "prompts": ["test prompt"],
                    "seeds": [10],
                    "runtime": {"device": "cuda", "precision": "float16"},
                    "output_root": str(Path(temp_dir) / "out"),
                }),
                encoding="utf-8",
            )
            exit_code = search_cli_main(["ecad", "--config", str(cfg_path), "--dry-run"])
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()

