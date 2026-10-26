"""Descriptive accommodation comparisons in the frozen October 2025 sample.

Product definitions and fixed borough contrasts are encoded explicitly; the
analysis does not select comparisons by their prices or interval significance.
This is exploratory analysis, not a preregistered confirmatory study. Minimum
stay means the *advertised minimum*, not an observed stay. No price correction,
tail trimming, occupancy or revenue inference is applied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRODUCTS = {
    "entire_1br_2guests": {
        "room_type": "Entire home/apt", "capacity_min": 2, "capacity_max": 2,
        "comparison_kind": "main_exact_capacity",
    },
    "entire_1br_2to4guests": {
        "room_type": "Entire home/apt", "capacity_min": 2, "capacity_max": 4,
        "comparison_kind": "capacity_sensitivity",
    },
    "private_1br_1to2guests": {
        "room_type": "Private room", "capacity_min": 1, "capacity_max": 2,
        "comparison_kind": "main_private_room",
    },
}
BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
STAY_SCOPES = ["all", "minimum_le7", "minimum_8to29", "minimum_ge30"]
ACCESS_THRESHOLDS_M = [500, 800]
SEED = 42


def _stay_filter(data: pd.DataFrame, scope: str) -> pd.Series:
    if scope == "all":
        return pd.Series(True, index=data.index)
    if scope == "minimum_le7":
        return data["minimum_nights"].between(1, 7)
    if scope == "minimum_8to29":
        return data["minimum_nights"].between(8, 29)
    if scope == "minimum_eq30":
        return data["minimum_nights"].eq(30)
    if scope == "minimum_ge30":
        return data["minimum_nights"] >= 30
    raise ValueError(f"Unknown minimum-night scope: {scope}")


def _product_filter(data: pd.DataFrame, specification: dict) -> pd.Series:
    return (data["room_type"].eq(specification["room_type"])
            & data["accommodates"].between(specification["capacity_min"], specification["capacity_max"])
            & data["bedrooms"].eq(1))


def _stats(data: pd.DataFrame) -> dict:
    price = data["price"]
    return {
        "n": len(data),
        "median_price_usd": float(price.median()) if len(data) else np.nan,
        "price_p25_usd": float(price.quantile(0.25)) if len(data) else np.nan,
        "price_p75_usd": float(price.quantile(0.75)) if len(data) else np.nan,
        "median_advertised_minimum_nights": float(data["minimum_nights"].median()) if len(data) else np.nan,
        "median_distance_m": float(data["dist_km"].median() * 1000) if len(data) else np.nan,
        "tract_count": int(data["GEOID"].nunique()),
    }


def _bootstrap_difference(reference: pd.DataFrame, comparison: pd.DataFrame,
                          replicates: int, rng: np.random.Generator) -> dict:
    """Percentile interval for a difference between two listing-weighted medians.

    Independently resample census tracts within each borough, retaining every
    selected listing from a drawn tract. This preserves dependence within a
    tract, but does not solve dependence between nearby tracts or missingness.
    """
    reference_clusters = [group["price"].to_numpy() for _, group in reference.groupby("GEOID", sort=True)]
    comparison_clusters = [group["price"].to_numpy() for _, group in comparison.groupby("GEOID", sort=True)]
    if min(len(reference_clusters), len(comparison_clusters)) < 2:
        raise ValueError("Tract bootstrap requires at least two tracts in each borough")
    differences = np.empty(replicates)
    percentages = np.empty(replicates)
    for index in range(replicates):
        reference_draw = rng.integers(0, len(reference_clusters), size=len(reference_clusters))
        comparison_draw = rng.integers(0, len(comparison_clusters), size=len(comparison_clusters))
        reference_median = np.median(np.concatenate([reference_clusters[i] for i in reference_draw]))
        comparison_median = np.median(np.concatenate([comparison_clusters[i] for i in comparison_draw]))
        differences[index] = comparison_median - reference_median
        percentages[index] = 100 * (comparison_median / reference_median - 1)
    return {
        "ci95_lower_usd": float(np.quantile(differences, 0.025)),
        "ci95_upper_usd": float(np.quantile(differences, 0.975)),
        "ci95_lower_pct": float(np.quantile(percentages, 0.025)),
        "ci95_upper_pct": float(np.quantile(percentages, 0.975)),
    }


def run_segment_analysis(root: Path | str, *, bootstrap_replicates: int = 1000) -> dict:
    """Save study-scope, comparable-product and fixed borough-comparison tables.

    Borough comparisons require >=100 listings per group; neighborhood rows
    require >=50. Main comparisons hold recorded room type, one bedroom,
    declared capacity, advertised minimum exactly 30 nights and station distance
    <=500 m fixed. >=30-night minimums, <=800 m and capacity 2–4 are sensitivity strata.
    Every result is historical and observational, without a booking or legal
    eligibility claim. The capacity field is not observed guest occupancy.
    """
    if bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    root = Path(root)
    data = pd.read_csv(root / "data" / "processed" / "listings.csv",
                       dtype={"id": str, "GEOID": str})
    required = {"id", "price", "room_type", "accommodates", "bedrooms", "minimum_nights",
                "dist_km", "borough", "neighbourhood_cleansed", "GEOID"}
    if not required.issubset(data):
        raise ValueError(f"Missing segmentation inputs: {sorted(required.difference(data))}")
    if not data["id"].is_unique or data["GEOID"].isna().any():
        raise ValueError("Segmentation requires distinct listing IDs and matched tract identifiers")
    if not (np.isfinite(data["price"]).all() and (data["price"] > 0).all()):
        raise ValueError("Segmentation requires finite positive advertised prices")
    if not data["minimum_nights"].ge(1).all():
        raise ValueError("Minimum nights must be present and at least one")
    if not np.isfinite(data["dist_km"]).all() or not data["dist_km"].ge(0).all():
        raise ValueError("Distance must be finite and nonnegative")
    tables = root / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    def save(name: str, rows: list[dict]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        destination = tables / f"segment_{name}.csv"
        frame.to_csv(destination, index=False, float_format="%.8f")
        saved[name] = str(destination.relative_to(root))
        return frame

    scope_rows = []
    for scope in STAY_SCOPES:
        subset = data[_stay_filter(data, scope)]
        scope_rows.append({"stay_scope": scope, "primary_population_n": len(data),
                           "share_of_primary_pct": 100 * len(subset) / len(data), **_stats(subset)})
    scope_table = save("scope", scope_rows)
    assert int(scope_table.loc[scope_table["stay_scope"] != "all", "n"].sum()) == len(data)
    threshold_rows = []
    for threshold_name, mask in [
        ("minimum_le7", data["minimum_nights"].between(1, 7)),
        ("minimum_le29", data["minimum_nights"].between(1, 29)),
        ("minimum_ge30", data["minimum_nights"] >= 30),
    ]:
        subset = data[mask]
        threshold_rows.append({"minimum_nights_threshold": threshold_name,
                               "primary_population_n": len(data),
                               "share_of_primary_pct": 100 * len(subset) / len(data),
                               **_stats(subset)})
    save("minimum_nights_thresholds", threshold_rows)

    room_rows = []
    for room, room_data in data.groupby("room_type", sort=True):
        for scope in STAY_SCOPES:
            stay_data = room_data[_stay_filter(room_data, scope)]
            for threshold in [None, *ACCESS_THRESHOLDS_M]:
                subset = stay_data if threshold is None else stay_data[stay_data["dist_km"] <= threshold / 1000]
                room_rows.append({
                    "room_type": room, "stay_scope": scope,
                    "maximum_station_distance_m": threshold,
                    "room_type_population_n": len(room_data), "room_stay_population_n": len(stay_data),
                    "share_of_room_stay_pct": 100 * len(subset) / len(stay_data) if len(stay_data) else np.nan,
                    **_stats(subset),
                })
    save("room_stay_access", room_rows)

    product_rows, borough_rows, neighborhood_rows = [], [], []
    for product_id, specification in PRODUCTS.items():
        capacity_pool = data[data["room_type"].eq(specification["room_type"])
                             & data["accommodates"].between(specification["capacity_min"], specification["capacity_max"])]
        product_data = data[_product_filter(data, specification)]
        product_rows.append({
            "product_id": product_id, **specification, "bedrooms_required": 1,
            "room_and_capacity_n": len(capacity_pool),
            "bedrooms_missing_excluded": int(capacity_pool["bedrooms"].isna().sum()),
            "other_bedroom_counts_excluded": int((capacity_pool["bedrooms"].notna()
                                                   & capacity_pool["bedrooms"].ne(1)).sum()),
            **_stats(product_data),
        })
        for scope in ["minimum_le7", "minimum_eq30", "minimum_ge30"]:
            for threshold in ACCESS_THRESHOLDS_M:
                selected = product_data[_stay_filter(product_data, scope)
                                        & (product_data["dist_km"] <= threshold / 1000)]
                context = {"product_id": product_id,
                           "comparison_kind": specification["comparison_kind"],
                           "stay_scope": scope, "maximum_station_distance_m": threshold}
                for borough in BOROUGHS:
                    subset = selected[selected["borough"].eq(borough)]
                    borough_rows.append({**context, "borough": borough,
                                         "eligible_n100": len(subset) >= 100, **_stats(subset)})
                for (borough, neighborhood), subset in selected.groupby(
                        ["borough", "neighbourhood_cleansed"], sort=True):
                    if len(subset) >= 50:
                        neighborhood_rows.append({**context, "borough": borough,
                                                  "neighborhood": neighborhood,
                                                  "eligible_n50": True, **_stats(subset)})
    save("products", product_rows)
    save("borough", borough_rows)
    save("neighborhood", neighborhood_rows)

    def comparisons_for_scope(scope: str) -> list[dict]:
        rng = np.random.default_rng(SEED)
        comparisons = []
        for product_id, specification in PRODUCTS.items():
            product_data = data[_product_filter(data, specification)
                                & _stay_filter(data, scope) & (data["dist_km"] <= 0.5)]
            reference = product_data[product_data["borough"].eq("Manhattan")]
            for borough in ["Brooklyn", "Queens"]:
                comparison = product_data[product_data["borough"].eq(borough)]
                reference_median = float(reference["price"].median())
                comparison_median = float(comparison["price"].median())
                eligible = min(len(reference), len(comparison)) >= 100
                row = {
                    "product_id": product_id, "comparison_kind": specification["comparison_kind"],
                    "stay_scope": scope, "maximum_station_distance_m": 500,
                    "reference_borough": "Manhattan", "comparison_borough": borough,
                    "n_reference": len(reference), "n_comparison": len(comparison),
                    "tracts_reference": int(reference["GEOID"].nunique()),
                    "tracts_comparison": int(comparison["GEOID"].nunique()),
                    "median_reference_usd": reference_median, "median_comparison_usd": comparison_median,
                    "difference_usd": comparison_median - reference_median,
                    "pct_difference": 100 * (comparison_median / reference_median - 1),
                    "eligible_n100": eligible, "bootstrap_replicates": bootstrap_replicates if eligible else 0,
                    "bootstrap_seed": SEED,
                    "ci_method": "95% percentile, independent tract resampling within each borough",
                }
                if eligible:
                    row.update(_bootstrap_difference(reference, comparison, bootstrap_replicates, rng))
                comparisons.append(row)
        return comparisons

    comparison_table = save("borough_bootstrap", comparisons_for_scope("minimum_eq30"))
    save("borough_bootstrap_sensitivity", comparisons_for_scope("minimum_ge30"))
    return {
        "primary_n": len(data),
        "minimum_le7_n": int(scope_table.loc[scope_table["stay_scope"].eq("minimum_le7"), "n"].iloc[0]),
        "minimum_le29_n": int(data["minimum_nights"].between(1, 29).sum()),
        "minimum_ge30_n": int(scope_table.loc[scope_table["stay_scope"].eq("minimum_ge30"), "n"].iloc[0]),
        "minimum_ge30_pct": float(scope_table.loc[scope_table["stay_scope"].eq("minimum_ge30"), "share_of_primary_pct"].iloc[0]),
        "tables": saved,
        "comparisons": comparison_table.to_dict(orient="records"),
        "interpretation": (
            "Historical advertised prices in explicitly defined product/stay/access strata. "
            "Bedrooms missing or not equal to one are excluded from product comparisons. "
            "Capacity is declared capacity, not actual guest occupancy. Minimum nights is a listing rule. "
            "The main borough contrasts hold that rule exactly at 30 nights; >=30-night contrasts are separate sensitivities. "
            "not observed stay duration. Amenities, exact neighborhood composition and listing quality remain different. "
            "Bootstrap intervals preserve within-tract dependence but not dependence between tracts. "
            "Intervals are exploratory and are not adjusted for multiple comparisons. "
            "These comparisons are descriptive, not causal, booking recommendations or evidence of legal eligibility."
        ),
    }
