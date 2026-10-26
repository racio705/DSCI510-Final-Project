"""Apply the frozen listing/spatial preparation to a supplied local snapshot.

This module never downloads a listing snapshot. Callers select the input only
after locking their temporal evaluation protocol. Frozen ACS/TIGER and MTA
references are reused, so the evaluation does not acquire a new geography or
community-data vintage. Host IDs are returned separately for in-memory cohort
checks and never appear in the feature frame or in any file written here.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .data import (
    NYC_COUNTIES, accessibility_features, attach_tracts, fetch_sources,
    parse_price, read_acs_table, read_stations, sha256_file,
)
from .scope import primary_mask

SAFE_COLUMNS = [
    "id", "price", "latitude", "longitude", "room_type", "property_type",
    "neighbourhood_cleansed", "accommodates", "bedrooms", "beds", "bathrooms",
    "minimum_nights", "number_of_reviews", "reviews_per_month", "review_scores_rating",
    "calculated_host_listings_count", "availability_365", "instant_bookable",
]
NUMERIC_COLUMNS = [
    "accommodates", "bedrooms", "beds", "bathrooms", "minimum_nights",
    "number_of_reviews", "reviews_per_month", "review_scores_rating",
    "calculated_host_listings_count", "availability_365",
]
REQUIRED_COLUMNS = {
    "id", "host_id", "price", "latitude", "longitude", "room_type",
    "property_type", "minimum_nights",
}


def _frozen_references(root: Path) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict]:
    """Resolve verified cached inputs without permitting any network access."""
    paths, manifest = fetch_sources(root, offline=True)
    tracts = gpd.read_file(paths["tracts"])
    tracts = tracts[tracts["COUNTYFP"].isin(NYC_COUNTIES)].copy()
    tracts = tracts[["GEOID", "COUNTYFP", "ALAND", "geometry"]].to_crs("EPSG:4326")
    tracts["borough"] = tracts["COUNTYFP"].map(NYC_COUNTIES)
    income = read_acs_table(paths["income"], "B19013").rename(columns={
        "estimate": "median_household_income", "moe": "median_household_income_moe"})
    population = read_acs_table(paths["population"], "B01003").rename(columns={
        "estimate": "population", "moe": "population_moe"})
    tracts = tracts.merge(income, on="GEOID", how="left", validate="one_to_one")
    tracts = tracts.merge(population, on="GEOID", how="left", validate="one_to_one")
    tracts["population_density_km2"] = tracts["population"] / (tracts["ALAND"].replace(0, np.nan) / 1_000_000)
    tracts["income_10k"] = tracts["median_household_income"] / 10_000
    tracts["log_income"] = np.log(tracts["median_household_income"])
    stations, station_quality = read_stations(paths["subway"])
    return tracts, stations, {
        "references": {key: manifest["sources"][key] for key in ["subway", "income", "population", "tracts"]},
        "nyc_tracts": len(tracts), **station_quality,
    }


def _listing_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Mirror the existing listing parser; select no records by price magnitude."""
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Snapshot missing required columns: {sorted(missing)}")
    raw = raw.copy()
    raw["price"] = parse_price(raw["price"])
    quality = {
        "raw_listings": len(raw),
        "missing_or_invalid_price": int(raw["price"].isna().sum()),
        "nonpositive_price": int((raw["price"] <= 0).sum()),
        "duplicate_listing_ids_removed": int(raw["id"].duplicated().sum()),
        "missing_listing_ids": int(raw["id"].isna().sum()),
        "numeric_input_columns_missing": [name for name in NUMERIC_COLUMNS if name not in raw],
    }
    rows = raw.drop_duplicates("id", keep="first").copy()
    # This mapping is private, never a prediction feature, and is not persisted.
    host_lookup = rows.dropna(subset=["id"]).set_index("id")["host_id"]
    rows = rows[rows["id"].notna() & np.isfinite(rows["price"]) & (rows["price"] > 0)].copy()
    quality["positive_price_unique_listings"] = len(rows)
    for name in ["latitude", "longitude"]:
        rows[name] = pd.to_numeric(rows[name], errors="coerce")
    rows = rows[rows["latitude"].between(-90, 90) & rows["longitude"].between(-180, 180)].copy()
    quality["valid_coordinate_listings"] = len(rows)
    features = rows[[column for column in SAFE_COLUMNS if column in rows]].copy()
    for name in NUMERIC_COLUMNS:
        features[name] = pd.to_numeric(rows[name], errors="coerce") if name in rows else np.nan
        features[name] = features[name].where(features[name] >= 0)
    if "bathrooms_text" in rows:
        fallback = pd.to_numeric(rows["bathrooms_text"].astype("string").str.extract(
            r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
        features["bathrooms"] = features["bathrooms"].fillna(fallback)
    if "instant_bookable" in features:
        features["instant_bookable"] = features["instant_bookable"].map({"t": 1, "f": 0, True: 1, False: 0})
    for name in ["room_type", "property_type", "neighbourhood_cleansed"]:
        if name not in features:
            features[name] = "Unknown"
        features[name] = features[name].fillna("Unknown").astype(str)
    return features, host_lookup, quality


def prepare_snapshot(root: Path, listing_path: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Prepare eligible long-stay rows from an explicit local listing snapshot.

    Eligibility is the existing non-hospitality rule, matched NYC tract, finite
    positive asking price, valid coordinates, and advertised minimum >=30
    nights. No price-tail threshold or future-label-based selection is used.
    Feature columns and transformations match the historical pipeline.

    Returns ``(features, host_ids, quality)``. The returned host Series has the
    exact same RangeIndex/order as features. Missing host IDs are retained as
    missing and counted; a caller must not classify them as verified new hosts.
    This function writes no feature, host or label files.
    """
    root, listing_path = Path(root), Path(listing_path)
    if not listing_path.is_file():
        raise FileNotFoundError(f"Supplied snapshot does not exist: {listing_path}")
    raw = pd.read_csv(listing_path, low_memory=False, dtype={"id": str, "host_id": str})
    features, host_lookup, quality = _listing_features(raw)
    quality.update({
        "snapshot_filename": listing_path.name,
        "snapshot_sha256": sha256_file(listing_path),
        "snapshot_bytes": listing_path.stat().st_size,
        "cohort_definition": "Positive-priced NYC tract matches; exclude hotel/hostel/resort; minimum_nights >=30; no price-tail trimming",
        "frozen_reference_period": "ACS 2019-2023 and TIGER/Line 2023; original frozen MTA extract",
    })
    if "last_scraped" in raw:
        dates = pd.to_datetime(raw["last_scraped"], errors="coerce")
        quality["listing_scrape_date_min"] = str(dates.min().date()) if dates.notna().any() else None
        quality["listing_scrape_date_max"] = str(dates.max().date()) if dates.notna().any() else None
    tracts, stations, reference_quality = _frozen_references(root)
    quality.update(reference_quality)
    points = gpd.GeoDataFrame(features, geometry=gpd.points_from_xy(
        features["longitude"], features["latitude"]), crs="EPSG:4326")
    joined, join_quality = attach_tracts(points, tracts.drop(columns="COUNTYFP"))
    quality.update(join_quality)
    quality["unmatched_nyc_tract_listings"] = int(joined["GEOID"].isna().sum())
    joined = joined[joined["GEOID"].notna()].copy().reset_index(drop=True)
    quality["priced_spatial_listings"] = len(joined)
    non_hospitality = primary_mask(joined)
    quality["hospitality_excluded"] = int((~non_hospitality).sum())
    quality["non_hospitality_listings"] = int(non_hospitality.sum())
    quality["minimum_stay_missing_or_below30_excluded"] = int((non_hospitality & ~joined["minimum_nights"].ge(30)).sum())
    joined = joined[non_hospitality & joined["minimum_nights"].ge(30)].copy().reset_index(drop=True)
    if joined.empty:
        raise ValueError("No eligible long-stay NYC listings remained in supplied snapshot")
    accessibility = accessibility_features(joined, stations)
    for name in accessibility:
        joined[name] = accessibility[name]
    joined["log_price"] = np.log1p(joined["price"])
    joined["log_minimum_nights"] = np.log1p(joined["minimum_nights"])
    joined["log_review_count"] = np.log1p(joined["number_of_reviews"])
    joined["log_host_listings_count"] = np.log1p(joined["calculated_host_listings_count"])
    result = pd.DataFrame(joined.drop(columns="geometry")).sort_values("id", kind="stable").reset_index(drop=True)
    if not result["id"].is_unique:
        raise AssertionError("Snapshot preparation expanded listing IDs")
    hosts = result["id"].map(host_lookup).rename("host_id")
    if "host_id" in result or "host_name" in result:
        raise AssertionError("Host identity must not enter the feature frame")
    quality["eligible_long_stay_listings"] = len(result)
    quality["listings_missing_host_id"] = int(hosts.isna().sum())
    quality["listings_missing_income"] = int(result["median_household_income"].isna().sum())
    quality["missing_pct"] = {name: float(result[name].isna().mean() * 100)
                              for name in NUMERIC_COLUMNS + ["median_household_income"]}
    return result, hosts, quality
