"""Pure numerical checks with synthetic values, never scientific analysis."""

import math
import unittest

from ..contracts import WINDOWS, AnalysisError
from ..statistics import complete_summary, pearson, quadratic_fit


class QuadraticTests(unittest.TestCase):
    def test_known_positive_negative_zero_coefficients_and_windows(self):
        for coefficient in (7., -4., 0.):
            points = {a: coefficient*a*a for a in (.01, .02, .05, .1, .2, 1.)}
            points[.2] = 999999  # Larger alphas cannot influence either local fit.
            for window, expected in WINDOWS.items():
                with self.subTest(coefficient=coefficient, window=window):
                    result = quadratic_fit(points, window)
                    self.assertTrue(result["eligible"])
                    self.assertAlmostEqual(result["coefficient_c"], coefficient)
                    self.assertAlmostEqual(result["empirical_curvature_proxy_2c"], 2*coefficient)
                    self.assertEqual(result["valid_points"], len(expected))
                    self.assertEqual(result["fitting_alphas"], list(expected))
                    self.assertAlmostEqual(result["rss"], 0.)
                    self.assertAlmostEqual(result["rmse"], 0.)
                    if coefficient:
                        self.assertAlmostEqual(result["r_squared"], 1.)
                    else:
                        self.assertIsNone(result["r_squared"])

    def test_residual_rmse_and_centered_unclamped_r_squared(self):
        points = {.01: 1., .02: 1.1, .05: 1.05, .1: 2.}
        result = quadratic_fit(points, "primary")
        xs, ys = [.0001, .0004, .0025], [1., 1.1, 1.05]
        c = sum(x*y for x, y in zip(xs, ys))/sum(x*x for x in xs)
        rss = sum((y-c*x)**2 for x, y in zip(xs, ys))
        self.assertAlmostEqual(result["coefficient_c"], c)
        self.assertAlmostEqual(result["rss"], rss)
        self.assertAlmostEqual(result["rmse"], math.sqrt(rss/3))
        self.assertLess(result["r_squared"], 0)
        self.assertAlmostEqual(result["fitted_points"][0]["residual"], 1-c*.0001)

    def test_constant_observations_have_explicit_undefined_r_squared(self):
        result = quadratic_fit({a: 1. for a in WINDOWS["primary"]}, "primary")
        self.assertIsNone(result["r_squared"])
        self.assertEqual(result["r_squared_status"], "undefined_constant_observed_delta_loss")

    def test_missing_window_point_never_becomes_a_shorter_fit(self):
        result = quadratic_fit({.01: 1., .02: None, .05: 2.}, "primary")
        self.assertFalse(result["eligible"])
        self.assertEqual(result["valid_points"], 2)
        self.assertIn("0.02", result["exclusion_reason"])
        self.assertIsNone(result["coefficient_c"])
        self.assertTrue(quadratic_fit({.01: 1., .02: 2., .05: 3.}, "primary")["eligible"])
        self.assertFalse(quadratic_fit({.01: 1., .02: 2., .05: 3.}, "secondary")["eligible"])

    def test_finite_validation(self):
        with self.assertRaises(AnalysisError):
            quadratic_fit({a: math.nan for a in WINDOWS["primary"]}, "primary")

    def test_seed_aggregation_uses_sample_sd_and_complete_sets(self):
        result = complete_summary([1., 2., 3.])
        self.assertEqual((result["mean"], result["sample_sd"], result["eligible_count"]), (2., 1., 3))
        self.assertIsNone(complete_summary([1., None, 3.])["mean"])
        self.assertIsNone(complete_summary([1., 2., 3.], compatible=False)["mean"])

    def test_correlations_are_descriptive_and_handle_degeneracy(self):
        self.assertAlmostEqual(pearson([(1., 3.), (2., 2.), (3., 1.)])["pearson_r"], -1.)
        self.assertIsNone(pearson([(1., 2.)]*3)["pearson_r"])
        self.assertEqual(pearson([(1., 2.)])["status"], "fewer_than_three_pairs")
