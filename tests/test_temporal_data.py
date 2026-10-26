"""Snapshot cleaning checks, using synthetic rows rather than future labels."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyc_airbnb.temporal_data import _listing_features


class TemporalDataTests(unittest.TestCase):
    def test_price_selection_preserves_tail_and_host_mapping_is_separate(self):
        raw = pd.DataFrame({
            "id": ["a", "a", "b", "c", "d"], "host_id": ["h1", "h2", "h3", "h4", None],
            "price": ["$50,000", "$100", "bad", "$0", "$80"],
            "latitude": [40.7] * 5, "longitude": [-74.] * 5,
            "room_type": ["Private room"] * 5, "property_type": ["Room in rental unit"] * 5,
            "minimum_nights": [30] * 5,
        })
        features, hosts, quality = _listing_features(raw)
        self.assertEqual(features["id"].tolist(), ["a", "d"])
        self.assertEqual(features["price"].tolist(), [50000., 80.])
        self.assertEqual(hosts.loc["a"], "h1")
        self.assertTrue(pd.isna(hosts.loc["d"]))
        self.assertNotIn("host_id", features)
        self.assertEqual(quality["duplicate_listing_ids_removed"], 1)

    def test_bathroom_fallback_and_missing_controls_match_historical_parser(self):
        raw = pd.DataFrame({
            "id": ["a"], "host_id": ["h1"], "price": [100],
            "latitude": [40.7], "longitude": [-74.], "room_type": ["Entire home/apt"],
            "property_type": ["Entire rental unit"], "minimum_nights": [30],
            "bathrooms": [np.nan], "bathrooms_text": ["1.5 baths"],
        })
        features, _, quality = _listing_features(raw)
        self.assertEqual(features["bathrooms"].iloc[0], 1.5)
        self.assertTrue(features["review_scores_rating"].isna().all())
        self.assertIn("review_scores_rating", quality["numeric_input_columns_missing"])


if __name__ == "__main__":
    unittest.main()
