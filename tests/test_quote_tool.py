"""Validate quote scope, feature integrity and decision-status propagation."""
import sys
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nyc_airbnb.quote_tool import PRIMARY, PROTOCOL_COMMIT, build_quote_features, quote_requests, validate_request, subgroup_diagnostic


def request():
    return {"latitude": 40.69, "longitude": -73.95, "room_type": "Entire home/apt",
            "property_type": "Entire rental unit", "accommodates": 2,
            "bedrooms": 1, "bathrooms": 1, "minimum_nights": 30, "hypothetical": True}


def references():
    tracts = gpd.GeoDataFrame({"GEOID": ["36047000100"], "borough": ["Brooklyn"],
                               "log_income": [np.log(80000)], "population_density_km2": [10000]},
                              geometry=[box(-74.1, 40.6, -73.8, 40.9)], crs=4326)
    stations = pd.DataFrame({"latitude": [40.69], "longitude": [-73.95], "station_name": ["Test station"]})
    return tracts, stations, {}


class FixedLogModel:
    def predict(self, frame):
        assert "price" not in frame and "log_price" not in frame
        return np.repeat(np.log1p(200), len(frame))


class FixedComparables:
    training_n_, training_hosts_ = 50, 20

    def predict(self, frame):
        return pd.DataFrame({"comparable_listing_median": [180.] * len(frame),
                             "comparable_host_median": [170.] * len(frame),
                             "comparable_level": [9] * len(frame)}, index=frame.index)


class QuoteToolTests(unittest.TestCase):
    def test_subgroup_precision_failure_is_visible(self):
        table = pd.DataFrame([{"model": PRIMARY, "borough": "Queens", "room_type": "Private room",
                               "n": 38, "mae_usd": 72., "median_ape_pct": 32.9, "coverage_80_pct": 78.9}])
        diagnostic = subgroup_diagnostic(table, "Queens", "Private room")
        self.assertEqual(diagnostic["n"], 38)
        self.assertTrue(any("30%" in text for text in diagnostic["flags"]))

    def test_target_price_and_out_of_scope_product_or_stay_are_rejected(self):
        for change in [{"price": 100}, {"Price": 100}, {"minimum_nights": 29},
                       {"room_type": "Hotel room"}, {"property_type": "Room in hostel"}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_request({**request(), **change})

    def test_invalid_counts_and_coordinates_are_rejected(self):
        for change in [{"latitude": 91}, {"longitude": -181}, {"accommodates": 0},
                       {"bedrooms": 1.5}, {"beds": -1}, {"bathrooms": float("inf")}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_request({**request(), **change})

    def test_geographic_enrichment_never_fabricates_a_target(self):
        frame, metadata = build_quote_features(Path("."), request(), references=references())
        self.assertEqual(frame["borough"].iloc[0], "Brooklyn")
        self.assertAlmostEqual(frame["dist_km"].iloc[0], 0.)
        self.assertEqual(frame["neighbourhood_cleansed"].iloc[0], "Unknown")
        self.assertFalse({"price", "log_price"}.intersection(frame))
        self.assertTrue(metadata["hypothetical"])
        self.assertIn("review_scores_rating", metadata["optional_inputs_missing"])

    def test_valid_earth_coordinates_outside_nyc_are_rejected(self):
        outside = {**request(), "latitude": 37.77, "longitude": -122.42}
        with self.assertRaisesRegex(ValueError, "NYC census tract"):
            build_quote_features(Path("."), outside, references=references())

    def test_failed_adoption_decision_remains_visible(self):
        bundle = {"models": {PRIMARY: FixedLogModel()}, "baselines": FixedComparables(),
                  "radii": {PRIMARY: .5, "comparable_listing_median": .4, "comparable_host_median": .3},
                  "protocol_commit": PROTOCOL_COMMIT, "decision": "criteria_not_met"}
        result = quote_requests(Path("."), [request()], bundle=bundle, references=references())[0]
        self.assertEqual(result["adoption_decision"], "criteria_not_met")
        self.assertEqual(result["point_quote_usd_per_night"], 200.)
        self.assertEqual(result["comparable_fallback"]["level"], 9)
        self.assertIn("did not establish", result["adoption_decision_note"])
        interval = result["estimates"][PRIMARY]["nominal_prediction_interval_80pct"]
        self.assertLess(interval["lower_usd"], 200.)
        self.assertGreater(interval["upper_usd"], 200.)
        self.assertIn("not guaranteed", result["interval_note"])


if __name__ == "__main__":
    unittest.main()
