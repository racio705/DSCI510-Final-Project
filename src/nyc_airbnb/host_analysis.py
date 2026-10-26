"""Anonymized host-account composition and supplemental host-group validation.

Raw host identifiers are used only in memory for aggregation and fold grouping.
They are never model features or exported. This same-snapshot evaluation tests
transfer to held-out hosts, not combined spatial, temporal or causal validity.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .data import SOURCE_SPECS, sha256_file
from .modeling import BASE_NUMERIC, CONTEXT_NUMERIC, TRANSIT_NUMERIC, CATEGORICAL, feature_pipeline, scores


MODEL_NAMES = ["Room + borough median", "Full without transit", "Full with transit"]


def load_host_rows(root: Path) -> pd.DataFrame:
    """Join every primary listing to its host from the exact pinned raw source."""
    root = Path(root)
    raw_path = root / "data" / "raw" / SOURCE_SPECS["airbnb"]["filename"]
    if sha256_file(raw_path) != SOURCE_SPECS["airbnb"]["sha256"]:
        raise ValueError("Raw Airbnb file differs from the pinned source")
    primary = pd.read_csv(root / "data" / "processed" / "listings.csv", dtype={"id": str, "GEOID": str})
    hosts = pd.read_csv(raw_path, usecols=["id", "host_id"], dtype=str)
    if not primary["id"].is_unique or not hosts["id"].is_unique:
        raise ValueError("Host linkage requires unique listing IDs in both inputs")
    joined = primary.merge(hosts, on="id", how="left", validate="one_to_one", indicator=True)
    if len(joined) != len(primary) or not joined["_merge"].eq("both").all() or joined["host_id"].isna().any():
        raise ValueError("Every primary listing must match exactly one nonmissing raw host")
    if not np.allclose(joined["log_price"], np.log1p(joined["price"]), atol=1e-12):
        raise ValueError("Host validation requires a consistent log1p(price) target")
    return joined.drop(columns="_merge").reset_index(drop=True)


def host_group_folds(data: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create disjoint host groups and require one held-out assignment per row."""
    if data["host_id"].isna().any() or data["host_id"].nunique() < n_splits:
        raise ValueError("Host folds require nonmissing IDs and enough distinct hosts")
    splits = list(GroupKFold(n_splits).split(data, groups=data["host_id"]))
    coverage = np.zeros(len(data), dtype=int)
    for train, test in splits:
        if not set(data.iloc[train]["host_id"]).isdisjoint(data.iloc[test]["host_id"]):
            raise AssertionError("Host appears in both training and held-out rows")
        coverage[test] += 1
    if not np.all(coverage == 1):
        raise AssertionError("Every listing must have exactly one held-out host prediction")
    return splits


def concentration_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Publish aggregate concentrations without names, IDs or identity hashes."""
    rows = []

    def summarize(subset: pd.DataFrame, scope: str, borough: str = "All", minimum_nights: int | None = None):
        counts = subset["host_id"].value_counts()
        without_largest = subset[subset["host_id"].ne(counts.index[0])] if len(counts) else subset
        host_medians = subset.groupby("host_id")["price"].median()
        rows.append({
            "scope": scope, "borough": borough, "advertised_minimum_nights": minimum_nights,
            "n": len(subset), "median_price_usd": float(subset["price"].median()),
            "unique_hosts": int(subset["host_id"].nunique()),
            "largest_host_n": int(counts.iloc[0]) if len(counts) else 0,
            "largest_host_share_pct": float(100 * counts.iloc[0] / len(subset)) if len(subset) else np.nan,
            "n_excluding_largest_host": len(without_largest),
            "median_price_excluding_largest_host_usd": float(without_largest["price"].median()) if len(without_largest) else np.nan,
            "median_of_host_medians_usd": float(host_medians.median()) if len(host_medians) else np.nan,
        })

    summarize(data, "Primary sample")
    summarize(data[data["minimum_nights"].eq(31)], "Primary, minimum 31 nights", minimum_nights=31)
    product = (data["room_type"].eq("Entire home/apt") & data["bedrooms"].eq(1)
               & data["accommodates"].eq(2) & data["dist_km"].le(0.5))
    dominant_31_hosts = []
    for borough in ["Manhattan", "Brooklyn", "Queens"]:
        for minimum in [30, 31]:
            subset = data[product & data["borough"].eq(borough) & data["minimum_nights"].eq(minimum)]
            summarize(subset, "Entire 1BR, capacity 2, subway within 500 m", borough, minimum)
            if minimum == 31 and len(subset):
                dominant_31_hosts.append(subset["host_id"].value_counts().index[0])
    same_dominant = len(dominant_31_hosts) == 3 and len(set(dominant_31_hosts)) == 1
    return pd.DataFrame(rows), bool(same_dominant)


def run_host_analysis(root: Path) -> dict:
    """Save anonymous host concentration and fixed-model host-group benchmarks."""
    root = Path(root)
    tables = root / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    data = load_host_rows(root)
    concentration, same_dominant = concentration_rows(data)
    concentration.to_csv(tables / "host_concentration.csv", index=False)
    pd.DataFrame([{
        "cohort": "Entire 1BR, capacity 2, subway within 500 m, minimum exactly 31 nights",
        "boroughs_compared": "Manhattan;Brooklyn;Queens",
        "same_dominant_host_across_boroughs": same_dominant,
    }]).to_csv(tables / "host_composition_relationship.csv", index=False)

    for column in CATEGORICAL:
        data[column] = data[column].fillna("Unknown").astype(object)
    target, actual = data["log_price"].to_numpy(), data["price"].to_numpy()
    predictions = {name: np.full(len(data), np.nan) for name in MODEL_NAMES}
    fold_ids = np.full(len(data), -1)
    fold_rows = []
    for fold, (train, test) in enumerate(host_group_folds(data), start=1):
        print(f"Host-group validation, fold {fold}/5", flush=True)
        fold_ids[test] = fold
        training, heldout = data.iloc[train], data.iloc[test]
        for name in MODEL_NAMES:
            if name == "Room + borough median":
                medians = training.groupby(["room_type", "borough"], observed=True)["log_price"].median()
                keys = pd.MultiIndex.from_frame(heldout[["room_type", "borough"]])
                predicted = medians.reindex(keys).fillna(np.median(target[train])).to_numpy()
            else:
                numeric = BASE_NUMERIC + CONTEXT_NUMERIC
                if name == "Full with transit":
                    numeric += TRANSIT_NUMERIC
                if "host_id" in numeric + CATEGORICAL:
                    raise AssertionError("Host identifier cannot be a prediction feature")
                model = feature_pipeline(numeric, CATEGORICAL, nonlinear=True)
                model.fit(training, target[train])
                predicted = model.predict(heldout)
            predictions[name][test] = predicted
            fold_rows.append({
                "validation": "Host-group 5-fold", "fold": fold, "model": name,
                "n_train": len(train), "n_test": len(test),
                "training_hosts": int(training["host_id"].nunique()),
                "heldout_hosts": int(heldout["host_id"].nunique()),
                **scores(actual[test], predicted),
            })
    metric_rows = []
    for name, prediction in predictions.items():
        if not np.isfinite(prediction).all() or not (fold_ids > 0).all():
            raise AssertionError("Host validation produced missing or repeated predictions")
        metric_rows.append({"validation": "Host-group 5-fold", "model": name, **scores(actual, prediction)})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(tables / "host_model_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(tables / "host_fold_metrics.csv", index=False)
    # Optional audit artifact is listing-level, ignored by Git, and also omits
    # the raw host identifiers used in memory for fold assignment.
    audit = data[["id", "price", "log_price"]].copy()
    audit["fold"] = fold_ids
    audit["predicted_log_room_borough"] = predictions["Room + borough median"]
    audit["predicted_log_no_transit"] = predictions["Full without transit"]
    audit["predicted_log_full"] = predictions["Full with transit"]
    audit.to_csv(root / "data" / "processed" / "host_oof_predictions.csv.gz", index=False)
    return {
        "primary_n": len(data), "unique_hosts": int(data["host_id"].nunique()),
        "raw_listing_join_coverage_pct": 100.0,
        "raw_source_sha256_verified": True,
        "same_dominant_host_in_three_borough_31night_cohorts": same_dominant,
        "concentration": concentration.replace({np.nan: None}).to_dict(orient="records"),
        "models": metrics.to_dict(orient="records"),
        "interpretation": [
            "Every host is entirely in training or held-out rows for a fold, and every listing has one held-out prediction.",
            "Host identifiers are never model features or exported; portfolio size remains a listing metadata feature.",
            "Minimum stay is an advertised rule; a 30-versus-31-night price difference is not a causal duration effect.",
            "Operator composition, property quality and geography differ across these groups.",
            "Median of host medians gives each distinct host equal weight; it is a different estimand from a listing-weighted price median.",
            "Removing each cohort's largest host is a descriptive concentration sensitivity, not a matched comparison or a causal estimate.",
            "Host-group validation tests held-out hosts within the same historical snapshot, not combined spatial/temporal independence.",
            "This supplemental evaluation was added after observing operator concentration; it is exploratory internal validation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Anonymized host composition and host-group cross-validation")
    parser.parse_args()
    summary = run_host_analysis(Path(__file__).resolve().parents[2])
    print(pd.DataFrame(summary["models"]).to_string(index=False))
    print(pd.DataFrame(summary["concentration"]).to_string(index=False))


if __name__ == "__main__":
    main()
