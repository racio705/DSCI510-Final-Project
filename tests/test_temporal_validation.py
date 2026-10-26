import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyc_airbnb.temporal_validation import conformal_radius, prediction_bounds, quote_metrics, paired_intervals


class TemporalValidationTests(unittest.TestCase):
    def test_finite_sample_quantile_uses_rank_not_interpolation(self):
        radius = conformal_radius(np.array([1., 3., 7., 15.]), np.ones(4))
        self.assertAlmostEqual(radius, np.log(8))
        lower, upper = prediction_bounds(np.array([1.]), radius)
        self.assertEqual(float(lower[0]), 0)
        self.assertAlmostEqual(float(upper[0]), 15)

    def test_percentage_metrics_keep_price_scale_and_inclusive_boundary(self):
        measured = quote_metrics(np.array([100., 200.]), np.array([125., 150.]), .5)
        self.assertAlmostEqual(measured["mae_usd"], 37.5)
        self.assertAlmostEqual(measured["median_ape_pct"], 25)
        self.assertEqual(measured["within_25pct"], 100)

    def test_empty_primary_cohort_retains_decision_schema(self):
        intervals = paired_intervals(np.array([]), {})
        self.assertTrue(intervals.empty)
        self.assertIn("contrast", intervals.columns)


if __name__ == "__main__":
    unittest.main()
