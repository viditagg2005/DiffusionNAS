"""CPU tests for the v3 architecture + policy search space and its two optimizers."""
from collections import Counter
import json
from pathlib import Path
import random
import tempfile
import unittest

from diffusionnas import nas_space
from diffusionnas.hypervolume import compute_hypervolume
from diffusionnas.nas_search import SEARCH_OBJECTIVES, run_nas_search
from diffusionnas.schema import validate_config
from diffusionnas.search_checkpoint import SearchPaused
from diffusionnas.search_space import policy_to_config

ROOT = Path(__file__).resolve().parents[1]


def config(method: str, budget: int = 24) -> dict:
    base = json.loads((ROOT / "configs" / "nas" / f"v3_{method}.json").read_text())
    base.update({"budget": budget, "seed": 5, "prompts": ["a cat"], "seeds": [1]})
    if method == "nsga2":
        base["population_size"] = 8
    return base


class NasSpaceTests(unittest.TestCase):
    def test_units_come_from_probe_selection(self):
        units = json.loads((ROOT / "configs" / "nas" / "units.json").read_text())["units"]
        self.assertEqual(sorted(nas_space.UNITS), sorted(units))
        self.assertNotIn("up3.attn2", nas_space.UNITS)  # catastrophic in the probe

    def test_random_genomes_decode_to_valid_run_configs(self):
        rng = random.Random(0)
        for _ in range(300):
            policy = nas_space.decode(nas_space.random_genome(rng))
            validate_config(policy_to_config(policy, ["a cat"], [1]))

    def test_number_of_removed_units_is_spread(self):
        rng = random.Random(1)
        counts = Counter(len(nas_space.decode(nas_space.random_genome(rng))["architecture"]["removed_units"])
                         for _ in range(3200))
        self.assertGreater(counts[0], 100)
        self.assertGreater(counts[len(nas_space.UNITS)], 100)

    def test_mutation_always_changes_execution_or_architecture(self):
        rng = random.Random(2)
        for _ in range(500):
            genome = nas_space.random_genome(rng)
            child = nas_space.mutate(genome, rng)
            self.assertNotEqual(nas_space.signature(nas_space.decode(genome)), nas_space.signature(nas_space.decode(child)))

    def test_signature_separates_architecture(self):
        rng = random.Random(3)
        genome = nas_space.random_genome(rng)
        other = list(genome)
        other[nas_space.PRECISION] = 1 - other[nas_space.PRECISION]
        self.assertNotEqual(nas_space.signature(nas_space.decode(genome)), nas_space.signature(nas_space.decode(other)))


class NasSearchTests(unittest.TestCase):
    def check(self, result, budget):
        self.assertEqual(result["evaluated_count"], budget)
        signatures = [nas_space.signature(c["policy"]) for c in result["candidates"]]
        self.assertEqual(len(signatures), len(set(signatures)))
        history = [h["hypervolume"] for h in result["hypervolume_history"]]
        self.assertTrue(all(b >= a - 1e-9 for a, b in zip(history, history[1:])))

    def test_random_and_nsga2_dry_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            for method in ("random", "nsga2"):
                result = run_nas_search(config(method), ROOT, Path(temp) / method, None, dry_run=True)
                self.check(result, 24)

    def test_nsga2_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            full = run_nas_search(config("nsga2", 20), ROOT, Path(temp) / "full", None, dry_run=True)
            with self.assertRaises(SearchPaused):
                run_nas_search(config("nsga2", 20), ROOT, Path(temp) / "resumed", None, dry_run=True, max_new_evaluations=11)
            resumed = run_nas_search(config("nsga2", 20), ROOT, Path(temp) / "resumed", None, dry_run=True)
            self.assertEqual([c["chromosome"] for c in resumed["candidates"]], [c["chromosome"] for c in full["candidates"]])

    def test_four_objective_hypervolume_respects_reference(self):
        ref = {"clip_score": 25.0, "clip_image_fidelity": 75.0, "latency_p50_ms": 600.0, "unet_weight_mb": 1700.0}
        inside = {"clip_score": 26.0, "clip_image_fidelity": 85.0, "latency_p50_ms": 500.0, "unet_weight_mb": 1600.0}
        below_floor = dict(inside, clip_score=24.0)
        self.assertAlmostEqual(compute_hypervolume([inside], ref, SEARCH_OBJECTIVES), 1 * 10 * 100 * 100)
        self.assertEqual(compute_hypervolume([below_floor], ref, SEARCH_OBJECTIVES), 0.0)


if __name__ == "__main__":
    unittest.main()


class SurrogateAssistedSearchTests(unittest.TestCase):
    def sa_config(self, method: str, budget: int = 40) -> dict:
        base = json.loads((ROOT / "configs" / "nas" / f"v3_{method}.json").read_text())
        base.update({"budget": budget, "seed": 2, "prompts": ["a cat"], "seeds": [1], "initial": 16, "batch": 8})
        return base

    def test_sa_arms_spend_budget_on_unique_policies(self):
        with tempfile.TemporaryDirectory() as temp:
            for method in ("sa_lhfid", "sa_nsga2"):
                result = run_nas_search(self.sa_config(method), ROOT, Path(temp) / method, None, dry_run=True)
                self.assertEqual(result["evaluated_count"], 40)
                signatures = [nas_space.signature(c["policy"]) for c in result["candidates"]]
                self.assertEqual(len(signatures), len(set(signatures)))

    def test_sa_resume_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as temp:
            full = run_nas_search(self.sa_config("sa_lhfid"), ROOT, Path(temp) / "full", None, dry_run=True)
            with self.assertRaises(SearchPaused):
                run_nas_search(self.sa_config("sa_lhfid"), ROOT, Path(temp) / "resumed", None, dry_run=True,
                               max_new_evaluations=21)
            resumed = run_nas_search(self.sa_config("sa_lhfid"), ROOT, Path(temp) / "resumed", None, dry_run=True)
            self.assertEqual([nas_space.signature(c["policy"]) for c in resumed["candidates"]],
                             [nas_space.signature(c["policy"]) for c in full["candidates"]])
