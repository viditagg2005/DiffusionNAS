"""CPU tests for continuous DTLZ problems, variation operators, the continuous MOEA loop and SA loop."""
import unittest

import numpy as np

from diffusionnas import continuous as c
from diffusionnas.sa_continuous import novel, run_sa


class ProblemTests(unittest.TestCase):
    def test_pareto_optimal_inputs_land_on_the_analytic_fronts(self):
        rng = np.random.default_rng(0)
        for m in (3, 6):
            x = rng.random((200, 10))
            x[:, m - 1:] = 0.5  # g = 0 for DTLZ1-4
            np.testing.assert_allclose(c.dtlz("dtlz1", x, m).sum(axis=1), 0.5, atol=1e-12)
            for name in ("dtlz2", "dtlz3", "dtlz4"):
                np.testing.assert_allclose(np.linalg.norm(c.dtlz(name, x, m), axis=1), 1.0, atol=1e-12)
            x[:, m - 1:] = 0.0  # g = 1 for DTLZ7
            f = c.dtlz("dtlz7", x, m)
            np.testing.assert_allclose(f[:, : m - 1], x[:, : m - 1])

    def test_reference_fronts(self):
        np.testing.assert_allclose(c.reference_front("dtlz1", 3).sum(axis=1), 0.5)
        np.testing.assert_allclose(np.linalg.norm(c.reference_front("dtlz2", 4), axis=1), 1.0)
        front7 = c.reference_front("dtlz7", 3, points=500)
        self.assertGreater(len(front7), 50)
        ideal, nadir = c.front_bounds(front7)
        self.assertTrue((nadir > ideal).all())

    def test_reference_vector_sizes_match_lhfid_paper(self):
        self.assertEqual([len(c.reference_vectors(m)) for m in (3, 5, 8, 10)], [105, 210, 156, 275])


class OperatorTests(unittest.TestCase):
    def test_operators_stay_in_unit_box(self):
        rng = np.random.default_rng(1)
        for _ in range(500):
            a, b = rng.random(12), rng.random(12)
            c1, c2 = c.sbx(a, b, 20.0, rng)
            y = c.polynomial_mutation(c1, 5.0, rng, rate=1.0)
            for v in (c1, c2, y):
                self.assertTrue(((v >= 0) & (v <= 1)).all())

    def test_run_keeps_population_and_counts_evaluations(self):
        v = c.reference_vectors(3)
        for algorithm in ("lhfid", "nsga3", "nsga2", "rvea"):
            r = c.run(algorithm, lambda x: c.dtlz("dtlz2", x, 3), 12, len(v), 5, v, seed=0)
            self.assertEqual(r["f"].shape, (len(v), 3))
            self.assertEqual(r["evaluations"], len(v) * 6)


class SaTests(unittest.TestCase):
    def test_novel_drops_duplicates(self):
        seen = np.array([[0.1, 0.2]])
        x = np.array([[0.1, 0.2], [0.3, 0.4], [0.3, 0.4]])
        self.assertEqual(novel(x, seen).tolist(), [1])

    def test_sa_uses_exact_budget(self):
        calls = []

        def observe(x):
            calls.append(len(x))
            return c.dtlz("dtlz2", x, 3)

        rng = np.random.default_rng(0)
        r = run_sa("lhfid", observe, rng.random((20, 6)), 47, seed=0, vectors=c.reference_vectors(3)[:28],
                   batch=10, inner_generations=3, surrogate="et")
        self.assertEqual(len(r["x"]), 47)
        self.assertEqual(sum(calls), 47)


if __name__ == "__main__":
    unittest.main()
