"""Validate frozen comparable hierarchy using synthetic training rows only."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyc_airbnb.temporal_baselines import ComparableBaselines, normalize_comparable_keys


def training_rows():
    return pd.DataFrame({
        "borough": ["Manhattan"] * 24, "neighbourhood_cleansed": ["Recorded area"] * 24,
        "room_type": ["Entire home/apt"] * 24, "accommodates": [2] * 24,
        "bedrooms": [1] * 24, "minimum_nights": [30] * 20 + [31] * 4,
        "price": [100.] * 20 + [200., 300., 400., 500.],
    })


class TemporalBaselineTests(unittest.TestCase):
    def test_first_cell_is_used_at_exact_listing_and_host_support_thresholds(self):
        training = training_rows()
        hosts = [name for name in ["A", "B", "C", "D", "E"] for _ in range(4)] + ["F", "G", "H", "I"]
        result = ComparableBaselines().fit(training, hosts).predict(training.iloc[[0]])
        self.assertEqual(result["comparable_level"].iloc[0], 1)
        self.assertEqual(result["comparable_listing_median"].iloc[0], 100.)
        self.assertEqual(result["comparable_host_median"].iloc[0], 100.)

    def test_sparse_host_cell_falls_through_even_with_twenty_listings(self):
        training = training_rows()
        baseline = ComparableBaselines().fit(training, ["A"] * 20 + ["B", "C", "D", "E"])
        prediction = baseline.predict(training.iloc[[0]])
        self.assertEqual(prediction["comparable_level"].iloc[0], 2)
        self.assertEqual(prediction["comparable_listing_median"].iloc[0], 100.)
        self.assertEqual(prediction["comparable_host_median"].iloc[0], 300.)

    def test_unseen_keys_follow_fixed_hierarchy_then_global(self):
        training = training_rows()
        baseline = ComparableBaselines().fit(training, ["A"] * 20 + ["B", "C", "D", "E"])
        query = pd.concat([training.iloc[[0]]] * 3, ignore_index=True)
        query.loc[0, "neighbourhood_cleansed"] = "Unseen area"
        query.loc[1, "borough"] = "Unseen borough"
        query.loc[2, "room_type"] = "Unseen room type"
        result = baseline.predict(query)
        self.assertEqual(result["comparable_level"].tolist(), [5, 8, 9])
        self.assertEqual(result["comparable_listing_median"].tolist(), [100., 100., 100.])
        self.assertEqual(result["comparable_host_median"].tolist(), [300., 300., 300.])

    def test_global_prediction_is_raw_dollar_median_not_log_median(self):
        training = training_rows().iloc[:2].copy()
        training["price"] = [100., 900.]
        result = ComparableBaselines().fit(training, ["A", "B"]).predict(training.iloc[[0]])
        self.assertEqual(result["comparable_listing_median"].iloc[0], 500.)
        self.assertEqual(result["comparable_host_median"].iloc[0], 500.)
        self.assertNotAlmostEqual(result["comparable_listing_median"].iloc[0], float(np.expm1(np.log1p(training.price).median())))
        self.assertEqual(result["comparable_level"].iloc[0], 9)

    def test_numeric_normalization_has_fixed_protocol_boundaries(self):
        query = pd.DataFrame({
            "accommodates": [1, 6, 7, 10, np.nan, 0, 2.5],
            "bedrooms": [0, 2, 3, 10, np.nan, -1, 1.5],
            "minimum_nights": [30, 31, 32, 59, 60, 89, 90],
        })
        keys = normalize_comparable_keys(query)
        self.assertEqual(keys.capacity.tolist(), ["1", "6", "7+", "7+", "Unknown", "Unknown", "Unknown"])
        self.assertEqual(keys.bedrooms.tolist(), ["0", "2", "3+", "3+", "Unknown", "Unknown", "Unknown"])
        self.assertEqual(keys.minimum_stay.tolist(), ["30", "31", "32-59", "32-59", "60-89", "60-89", "90+"])

    def test_prediction_ignores_any_supplied_price_and_keeps_query_index(self):
        training = training_rows()
        baseline = ComparableBaselines().fit(training, ["A"] * 20 + ["B", "C", "D", "E"])
        query = training.iloc[[0]].copy()
        query.index = [1234]
        before = baseline.predict(query)
        query["price"] = 1e12
        pd.testing.assert_frame_equal(before, baseline.predict(query))
        self.assertEqual(before.index.tolist(), [1234])


if __name__ == "__main__":
    unittest.main()
