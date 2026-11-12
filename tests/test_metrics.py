"""CPU tests for KID/FID statistics and the paired prompt bootstrap."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diffusionnas.distribution import fid, kid
from diffusionnas.stats import paired_bootstrap, per_prompt_scores


class DistributionTests(unittest.TestCase):
    def test_kid_near_zero_for_same_distribution_and_positive_for_shift(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=(300, 16)), rng.normal(size=(300, 16))
        shifted = rng.normal(loc=0.5, size=(300, 16))
        same = kid(a, b, subsets=20)["mean_x1000"]
        different = kid(a, shifted, subsets=20)["mean_x1000"]
        self.assertLess(abs(same), different / 10)
        self.assertGreater(different, 0)

    def test_fid_zero_on_identical_sets(self):
        a = np.random.default_rng(1).normal(size=(200, 8))
        self.assertAlmostEqual(fid(a, a), 0.0, places=4)
        self.assertGreater(fid(a, a + 1.0), 7.9)


class BootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_on_constant_shift(self):
        reference = {i: float(i) for i in range(30)}
        candidate = {i: float(i) - 1.0 for i in range(30)}
        result = paired_bootstrap(candidate, reference, draws=500)
        self.assertAlmostEqual(result["mean_difference"], -1.0)
        self.assertAlmostEqual(result["ci95_low"], -1.0)
        self.assertAlmostEqual(result["ci95_high"], -1.0)

    def test_bootstrap_p_value_is_one_for_identical_runs_and_small_for_a_shift(self):
        scores = {i: float(i % 7) for i in range(60)}
        self.assertEqual(paired_bootstrap(scores, scores, draws=500)["p_value"], 1.0)
        shifted = {i: v + 1.0 + 0.1 * (i % 3) for i, v in scores.items()}
        self.assertLess(paired_bootstrap(shifted, scores, draws=500)["p_value"], 0.01)

    def test_per_prompt_scores_average_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            rows = [{"prompt_index": 0, "seed": 1, "clip_score": 20.0}, {"prompt_index": 0, "seed": 2, "clip_score": 22.0}]
            Path(temporary, "quality_measurements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(per_prompt_scores(temporary), {0: 21.0})


if __name__ == "__main__":
    unittest.main()
