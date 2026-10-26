"""Check unseen-host grouping and anonymized composition exports."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyc_airbnb.host_analysis import concentration_rows, host_group_folds


class HostAnalysisTests(unittest.TestCase):
    def test_host_folds_are_disjoint_and_cover_every_listing_once(self):
        data = pd.DataFrame({"host_id": ["A"] * 5 + ["B"] * 4 + ["C"] * 3 + ["D"] * 2 + ["E"] + ["F"]})
        coverage = np.zeros(len(data), dtype=int)
        for train, test in host_group_folds(data):
            self.assertTrue(set(data.iloc[train].host_id).isdisjoint(data.iloc[test].host_id))
            coverage[test] += 1
        np.testing.assert_array_equal(coverage, np.ones(len(data), dtype=int))

    def test_concentration_exports_aggregates_without_host_identifiers(self):
        data = pd.DataFrame({
            "host_id": ["SECRET_A", "SECRET_A", "SECRET_B", "SECRET_A"],
            "price": [100., 200., 300., 400.], "minimum_nights": [31, 31, 30, 31],
            "room_type": ["Entire home/apt"] * 4, "bedrooms": [1] * 4,
            "accommodates": [2] * 4, "dist_km": [0.1] * 4,
            "borough": ["Manhattan", "Brooklyn", "Brooklyn", "Queens"],
        })
        concentration, same = concentration_rows(data)
        overall = concentration.iloc[0]
        self.assertEqual(overall["largest_host_n"], 3)
        self.assertEqual(overall["largest_host_share_pct"], 75.)
        self.assertTrue(same)
        self.assertNotIn("host_id", concentration.columns)
        self.assertNotIn("SECRET_", concentration.to_csv(index=False))


if __name__ == "__main__":
    unittest.main()
