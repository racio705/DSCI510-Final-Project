"""Checks for data errors that can materially change the spatial conclusions."""

import tempfile
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nyc_airbnb.data import (
    accessibility_features, attach_tracts, clean_acs_values, fetch_sources, parse_price,
    read_acs_table, read_stations,
)


class DataTests(unittest.TestCase):
    def test_price_values_preserve_missingness(self):
        actual = parse_price(pd.Series(["$1,234.50", 75, "bad", None, "$0.00", -2]))
        self.assertEqual(actual.iloc[0], 1234.5)
        self.assertEqual(actual.iloc[1], 75)
        self.assertTrue(actual.iloc[2:4].isna().all())
        self.assertEqual(actual.iloc[4], 0)
        self.assertEqual(actual.iloc[5], -2)

    def test_census_sentinels_are_not_income(self):
        result = clean_acs_values(pd.Series(["-666666666", "-999999999", "0", "250001", "120000"]), positive_only=True)
        self.assertTrue(result.iloc[:3].isna().all())
        self.assertEqual(result.iloc[3], 250001)  # Valid upper-coded estimate.

    def test_metric_distance_agrees_with_haversine(self):
        point = gpd.GeoDataFrame(geometry=[Point(-73.99, 40.75)], crs=4326)
        stations = pd.DataFrame({"longitude": [-73.98], "latitude": [40.75], "station_name": ["A"]})
        actual_km = accessibility_features(point, stations)["dist_km"].iloc[0]
        lat = np.deg2rad(40.75)
        delta = np.deg2rad(0.01)
        haversine_km = 6371.0088 * 2 * np.arcsin(np.sqrt(np.cos(lat) ** 2 * np.sin(delta / 2) ** 2))
        self.assertLess(abs(actual_km / haversine_km - 1), 0.005)
        self.assertLess(actual_km, 0.85)  # EPSG:3857 incorrectly gives ~1.11 km.

    def test_tract_join_does_not_expand_boundary_points(self):
        tracts = gpd.GeoDataFrame({"GEOID": ["A", "B"], "income": [100, 200]},
                                 geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)], crs=4326)
        points = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5), Point(1, 0.5), Point(3, 3), Point(0, 0.5)], crs=4326)
        result, quality = attach_tracts(points, tracts)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.loc[0, "GEOID"], "A")
        self.assertTrue(pd.isna(result.loc[1, "GEOID"]))
        self.assertEqual(quality["ambiguous_boundary_listings"], 1)
        self.assertEqual(result.loc[3, "GEOID"], "A")
        self.assertEqual(quality["unique_boundary_matches"], 1)

    def test_station_counts_exclude_sir_and_duplicate_platforms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stations.csv"
            pd.DataFrame({
                "Station ID": [1, 1, 2, 3, 4], "Division": ["IRT", "IRT", "BMT", "SIR", "IND"],
                "Stop Name": ["A", "A", "B", "SIR", "C"],
                "GTFS Latitude": [40.75, 40.75, 40.75, 40.6, 40.76],
                "GTFS Longitude": [-73.99, -73.99, -73.99, -74.1, -73.98],
            }).to_csv(path, index=False)
            result, quality = read_stations(path)
        self.assertEqual(len(result), 2)
        self.assertEqual(quality["sir_rows_excluded"], 1)
        self.assertEqual(quality["station_ids_unique_after_sir_exclusion"], 3)

    def test_acs_national_file_restricts_to_nyc_tracts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "income.dat"
            path.write_text("GEO_ID|E001|M001\n1400000US36061000100|100000|5000\n"
                            "1400000US36001000100|60000|2000\n0500000US36061|90000|3000\n", encoding="utf-8")
            result = read_acs_table(path, "B19013")
        self.assertEqual(result["GEOID"].tolist(), ["36061000100"])
        self.assertEqual(result["moe"].iloc[0], 5000)

    def test_cached_source_change_fails_instead_of_silently_changing_results(self):
        spec = {"example": {"filename": "source.csv", "url": "https://example.invalid/source.csv",
                            "period": "test", "provenance": "test", "download": False}}
        with tempfile.TemporaryDirectory() as directory, patch("nyc_airbnb.data.SOURCE_SPECS", spec):
            root = Path(directory)
            raw = root / "data" / "raw"
            raw.mkdir(parents=True)
            source = raw / "source.csv"
            source.write_text("price\n100\n", encoding="utf-8")
            fetch_sources(root, offline=True)
            source.write_text("price\n200\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                fetch_sources(root, offline=True)

    def test_offline_run_copies_bundled_reference_without_network(self):
        spec = {"example": {"filename": "source.csv", "url": "https://example.invalid/source.csv",
                            "period": "test", "provenance": "test", "download": False,
                            "reference": "data/reference/source.csv"}}
        with tempfile.TemporaryDirectory() as directory, patch("nyc_airbnb.data.SOURCE_SPECS", spec):
            root = Path(directory)
            reference = root / "data" / "reference" / "source.csv"
            reference.parent.mkdir(parents=True)
            reference.write_text("price\n100\n", encoding="utf-8")
            with patch("nyc_airbnb.data._download") as download:
                paths, _ = fetch_sources(root, offline=True)
                download.assert_not_called()
            self.assertEqual(paths["example"].read_text(), reference.read_text())


if __name__ == "__main__":
    unittest.main()
