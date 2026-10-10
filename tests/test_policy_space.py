"""CPU-only tests for the method-aware v2 search space and deduplicating optimizers."""
from collections import Counter
import json
from pathlib import Path
import random
import tempfile
import unittest

from diffusionnas import policy_space as ps
from diffusionnas.ecad import run_ecad_search
from diffusionnas.random_search import run_random_search
from diffusionnas.schema import validate_config
from diffusionnas.search_checkpoint import SearchPaused
from diffusionnas.search_space import policy_to_config


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = {"device": "cuda", "precision": "float16", "warmup_runs": 0, "repetitions": 1}


def search_config(method: str, budget: int) -> dict:
    config = {
        "search_method": method,
        "search_space": "v2",
        "budget": budget,
        "seed": 3,
        "prompts": ["an astronaut"],
        "seeds": [7],
        "runtime": RUNTIME,
        "reference_point": {"clip_score": 20.0, "latency_p50_ms": 600.0, "peak_allocated_mb": 4000.0},
    }
    if method == "nsga2":
        config.update({"population_size": 6, "crossover_rate": 0.9, "mutation_rate": 1.0})
    return config


class PolicySpaceTests(unittest.TestCase):
    def test_random_genomes_decode_to_valid_configs(self):
        rng = random.Random(0)
        for _ in range(500):
            policy = ps.decode(ps.random_genome(rng))
            validate_config(policy_to_config(policy, ["a cat"], [1]))

    def test_methods_are_sampled_uniformly(self):
        rng = random.Random(1)
        counts = Counter(ps.decode(ps.random_genome(rng))["method"] for _ in range(3000))
        for method in ps.METHODS:
            self.assertAlmostEqual(counts[method] / 3000, 1 / 3, delta=0.04)

    def test_only_inert_deepcache_intervals_collide(self):
        # DeepCache with interval >= steps runs one full step whatever the interval; dedupe must merge them.
        rng = random.Random(2)
        groups: dict[str, set] = {}
        for _ in range(3000):
            genome = ps.random_genome(rng)
            groups.setdefault(ps.signature(ps.decode(genome)), set()).add(tuple(genome))
        for genomes in groups.values():
            if len(genomes) > 1:
                for genome in genomes:
                    policy = ps.decode(list(genome))
                    self.assertEqual(policy["method"], "deepcache")
                    self.assertGreaterEqual(policy["deepcache_interval"], policy["num_inference_steps"])

    def test_mutation_always_changes_execution(self):
        rng = random.Random(3)
        for _ in range(500):
            genome = ps.random_genome(rng)
            child = ps.mutate(genome, rng)
            self.assertNotEqual(ps.signature(ps.decode(genome)), ps.signature(ps.decode(child)))

    def test_uniform_flexiffusion_equal_to_deepcache_shares_signature(self):
        deepcache = {
            "method": "deepcache", "scheduler": "ddim", "num_inference_steps": 12, "guidance_scale": 7.5,
            "deepcache_interval": 3, "deepcache_branch_id": 3,
        }
        flex = {
            "method": "flexiffusion", "scheduler": "ddim", "num_inference_steps": 12, "guidance_scale": 7.5,
            "flexiffusion": {"segment_length": 3, "segments": [{"branch": 3, "interval": 3}] * 4},
        }
        self.assertEqual(ps.signature(deepcache), ps.signature(flex))

    def test_encode_decode_roundtrip_for_seeds(self):
        for genome in ps.initial_genomes():
            self.assertEqual(ps.encode(ps.decode(genome)), genome)


class DeduplicatingSearchTests(unittest.TestCase):
    def assert_unique(self, result):
        signatures = [ps.signature(candidate["policy"]) for candidate in result["candidates"]]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_random_v2_never_repeats_a_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_random_search(search_config("random", 40), ROOT, output_dir=Path(temp_dir), dry_run=True)
            self.assertEqual(result["evaluated_count"], 40)
            self.assertEqual(result["search_space"], "v2")
            self.assert_unique(result)

    def test_nsga2_v2_never_repeats_a_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_ecad_search(search_config("nsga2", 30), ROOT, output_dir=Path(temp_dir), dry_run=True)
            self.assertEqual(result["search_method"], "nsga2")
            self.assertEqual(result["evaluated_count"], 30)
            self.assert_unique(result)

    def test_nsga2_v2_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = search_config("nsga2", 14)
            full = run_ecad_search(config, ROOT, output_dir=root / "full", dry_run=True)
            with self.assertRaises(SearchPaused):
                run_ecad_search(config, ROOT, output_dir=root / "resumed", dry_run=True, max_new_evaluations=8)
            resumed = run_ecad_search(config, ROOT, output_dir=root / "resumed", dry_run=True)
            self.assertEqual(
                [c["chromosome"] for c in resumed["candidates"]], [c["chromosome"] for c in full["candidates"]]
            )
            self.assertEqual(resumed["hypervolume_history"], full["hypervolume_history"])
            checkpoint = json.loads((root / "resumed" / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["search_method"], "nsga2")


if __name__ == "__main__":
    unittest.main()
