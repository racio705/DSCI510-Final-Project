import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyc_airbnb.modeling import feature_pipeline, scores, moran_permutation
from nyc_airbnb.scope import primary_mask


class ModelingTests(unittest.TestCase):
    def test_private_room_hotel_is_excluded_without_price_filter(self):
        data = pd.DataFrame({"room_type": ["Private room", "Hotel room", "Entire home/apt", "Private room"],
                             "property_type": ["Room in hotel", "Unknown", "Entire home", "Room in rental unit"],
                             "price": [100, 100, 50000, 80]})
        self.assertEqual(primary_mask(data).tolist(), [False, False, True, True])

    def test_dollar_metrics_are_back_transformed(self):
        actual = np.array([100., 200., 300.])
        measured = scores(actual, np.log1p(actual + 10))
        self.assertAlmostEqual(measured["mae_usd"], 10)
        self.assertAlmostEqual(measured["rmse_usd"], 10)

    def test_test_values_do_not_change_imputation(self):
        train = pd.DataFrame({"x": [1., 3., np.nan], "room": ["A", "A", "B"]})
        model = feature_pipeline(["x"], ["room"])
        model.fit(train, np.array([1., 2., 3.]))
        median = model.named_steps["features"].named_transformers_["numeric"].named_steps["impute"].statistics_[0]
        model.predict(pd.DataFrame({"x": [1e9, np.nan], "room": ["unseen", "A"]}))
        self.assertEqual(median, 2.)
        self.assertEqual(model.named_steps["features"].named_transformers_["numeric"].named_steps["impute"].statistics_[0], 2.)

    def test_moran_is_seeded_and_reports_finite_values(self):
        coords = np.column_stack([np.arange(12), np.zeros(12)])
        a = moran_permutation(np.arange(12), coords, permutations=19)
        b = moran_permutation(np.arange(12), coords, permutations=19)
        self.assertEqual(a, b)
        self.assertTrue(np.isfinite(a["moran_i"]))
        self.assertGreaterEqual(a["p_two_sided"], .05)


if __name__ == "__main__":
    unittest.main()
