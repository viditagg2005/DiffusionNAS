"""CPU tests for the many-objective selection schemes (NSGA-II, NSGA-III, LHFiD) and indicators."""
import random
import unittest

import numpy as np

from diffusionnas import moea, nas_space


class GeometryTests(unittest.TestCase):
    def test_das_dennis_sizes_match_population(self):
        self.assertEqual(len(moea.das_dennis(6, 3)), 56)
        self.assertEqual(len(moea.das_dennis(4, 5)), 56)
        self.assertTrue(np.allclose(moea.das_dennis(6, 3).sum(axis=1), 1.0))

    def test_two_layer_inner_vectors_have_no_zero_components(self):
        vectors = moea.two_layer(6, 2, 2)
        self.assertEqual(len(vectors), 42)
        self.assertTrue((vectors[21:] > 0).all())
        self.assertTrue(np.isfinite(moea.lhfid_preferences(vectors, "inverse_w")).all())

    def test_perpendicular_distance_on_axis(self):
        d = moea.perpendicular_distance(np.array([[1.0, 1.0]]), np.array([[1.0, 0.0], [1.0, 1.0]]))
        self.assertAlmostEqual(d[0, 0], 1.0)
        self.assertAlmostEqual(d[0, 1], 0.0)


class HfDominanceTests(unittest.TestCase):
    def test_paper_table_ii_cluster_a_equal_preferences(self):
        # LHFiD paper, Fig. 2 / Table II: alpha = S2; with p_m = 1/M only S1 hf-dominates S2.
        s1, s2, s5 = np.array([0.625, 0.175, 0.150]), np.array([0.675, 0.225, 0.100]), np.array([0.625, 0.325, 0.080])
        equal = np.full(3, 1 / 3)
        self.assertTrue(moea.hf_dominates(s1, s2, equal))
        self.assertFalse(moea.hf_dominates(s5, s2, equal))

    def test_pareto_dominance_implies_hf_dominance(self):
        rng = np.random.default_rng(0)
        for _ in range(500):
            y = rng.random(5)
            x = y - rng.random(5) * (rng.random(5) < 0.6)
            if moea.dominates(x, y):
                self.assertTrue(moea.hf_dominates(x, y, rng.random(5) + 0.01))


class SelectionTests(unittest.TestCase):
    def test_every_selector_returns_n_distinct_members(self):
        rng = np.random.default_rng(1)
        f = rng.random((112, 6))
        vectors = moea.das_dennis(6, 3)
        for name, select in moea.SELECTORS.items():
            chosen = select(f, 56, {"vectors": vectors}, random.Random(0))
            self.assertEqual(len(chosen), 56, name)
            self.assertEqual(len(set(chosen)), 56, name)

    def test_selectors_keep_a_dominating_point(self):
        f = np.vstack([np.full((1, 4), 0.0), np.random.default_rng(2).random((111, 4)) + 0.1])
        for name, select in moea.SELECTORS.items():
            self.assertIn(0, select(f, 56, {"vectors": moea.das_dennis(4, 5)}, random.Random(0)), name)


class RunTests(unittest.TestCase):
    def test_all_algorithms_spend_budget_on_unique_policies(self):
        def evaluate(policies):
            return np.array([[len(p["architecture"]["removed_units"]), -p["num_inference_steps"],
                              p["guidance_scale"], float(p["architecture"]["weight_precision"] == "fp16")]
                             for p in policies], dtype=float)
        for algorithm in ("random", "nsga2", "nsga3", "lhfid", "lhfid_equal"):
            result = moea.run(algorithm, evaluate, 300, 56, moea.das_dennis(4, 5), seed=0)
            self.assertEqual(len(result["objectives"]), 300, algorithm)

    def test_monte_carlo_hypervolume_of_a_single_point(self):
        samples = np.random.default_rng(3).random((200000, 2))
        hv = moea.hypervolume_mc(np.array([[0.5, 0.5]]), np.array([1.0, 1.0]), np.zeros(2), samples)
        self.assertAlmostEqual(hv, 0.25, delta=0.005)


if __name__ == "__main__":
    unittest.main()
