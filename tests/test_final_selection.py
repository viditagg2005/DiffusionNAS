"""CPU tests for noise-aware final selection (src/diffusionnas/final_selection.py)."""
import unittest

import numpy as np

from diffusionnas import final_selection as fs


class FinalSelectionTests(unittest.TestCase):
    def setUp(self):
        self.ref = np.array([0.0, 0.0, 10.0, 10.0])

    def test_box_is_relative_to_reference(self):
        ref = fs.box(27.0, 500.0)
        np.testing.assert_allclose(ref, [-26.0, -75.0, 590.0, 1700.0])

    def test_out_of_box_points_are_never_chosen(self):
        pts = np.array([[-1, -1, 5, 5], [-2, -2, 11, 5], [-1, -1, 5, 10]])  # 2nd: latency out; 3rd: on boundary
        self.assertEqual(fs.greedy_hssp(pts, self.ref, 3), [0])

    def test_greedy_takes_largest_gain_first_and_is_incremental(self):
        pts = np.array([[-1, -1, 9, 9], [-3, -3, 2, 2], [-2, -1, 1, 9], [-1, -2, 9, 1]])
        order5 = fs.greedy_hssp(pts, self.ref, 4)
        self.assertEqual(order5[0], 1)
        self.assertEqual(fs.greedy_hssp(pts, self.ref, 2), order5[:2])  # naive k = prefix of screen order

    def test_fewer_in_box_than_k(self):
        pts = np.array([[-1, -1, 5, 5]])
        self.assertEqual(fs.greedy_hssp(pts, self.ref, 5), [0])

    def test_hypervolume_ignores_outside_points(self):
        a = fs.hypervolume(np.array([[-1, -1, 9, 9]]), self.ref)
        b = fs.hypervolume(np.array([[-1, -1, 9, 9], [-5, -5, 20, 0]]), self.ref)
        self.assertAlmostEqual(a, b)
        self.assertAlmostEqual(a, 1.0)

    def test_nondominated(self):
        pts = [[1, 1], [0, 2], [2, 2], [1, 1]]
        self.assertEqual(fs.nondominated_indices(pts), [0, 1, 3])


if __name__ == "__main__":
    unittest.main()
