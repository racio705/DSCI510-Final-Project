"""Guard the exact product definition and the direction of price contrasts."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyc_airbnb.segment_analysis import PRODUCTS, _bootstrap_difference, _product_filter


class SegmentAnalysisTests(unittest.TestCase):
    def test_exact_capacity_product_excludes_studios_missing_bedrooms_and_private_rooms(self):
        listings = pd.DataFrame({
            "room_type": ["Entire home/apt"] * 5 + ["Private room"],
            "accommodates": [2, 3, 4, 2, 2, 2],
            "bedrooms": [1, 1, 1, 0, np.nan, 1],
        })
        exact = _product_filter(listings, PRODUCTS["entire_1br_2guests"])
        broad = _product_filter(listings, PRODUCTS["entire_1br_2to4guests"])
        private = _product_filter(listings, PRODUCTS["private_1br_1to2guests"])
        self.assertEqual(listings.index[exact].tolist(), [0])
        self.assertEqual(listings.index[broad].tolist(), [0, 1, 2])
        self.assertEqual(listings.index[private].tolist(), [5])

    def test_constant_cluster_prices_give_degenerate_correctly_directed_difference(self):
        reference = pd.DataFrame({"GEOID": ["A", "A", "A", "B"], "price": [100., 100., 100., 100.]})
        comparison = pd.DataFrame({"GEOID": ["C", "D", "D"], "price": [60., 60., 60.]})
        result = _bootstrap_difference(reference, comparison, 100, np.random.default_rng(42))
        self.assertEqual(result["ci95_lower_usd"], -40.)
        self.assertEqual(result["ci95_upper_usd"], -40.)
        self.assertEqual(result["ci95_lower_pct"], -40.)
        self.assertEqual(result["ci95_upper_pct"], -40.)


if __name__ == "__main__":
    unittest.main()
