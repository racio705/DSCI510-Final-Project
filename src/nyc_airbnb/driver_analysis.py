"""Grouped held-out permutation analysis of the fixed full prediction model.

This measures a fitted model's predictive reliance on each feature group.
It does not identify causal price determinants. Correlation across groups can
share predictive information or make shuffled combinations unrealistic.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .modeling import BASE_NUMERIC, CONTEXT_NUMERIC, TRANSIT_NUMERIC, CATEGORICAL, SEED, feature_pipeline


FEATURE_GROUPS = {
    "Listing product": ["room_type", "property_type"],
    "Capacity": ["bedrooms", "bathrooms", "beds", "accommodates"],
    "Geography and tract context": ["latitude", "longitude", "borough", "log_income", "population_density_km2"],
    "Minimum stay": ["log_minimum_nights"],
    "Reviews and rating": ["log_review_count", "review_scores_rating"],
    "Host portfolio size": ["log_host_listings_count"],
    "Subway accessibility": ["dist_km", "stations_within_800m"],
}


def jointly_permute(frame: pd.DataFrame, columns: list[str], seed: int) -> pd.DataFrame:
    """Shuffle rows of a group jointly, preserving relationships within it."""
    order = np.random.default_rng(seed).permutation(len(frame))
    shuffled = frame.copy()
    for column in columns:
        shuffled[column] = frame[column].iloc[order].to_numpy(copy=True)
    return shuffled


def _errors(actual: np.ndarray, log_actual: np.ndarray, log_prediction: np.ndarray) -> dict:
    dollar_prediction = np.maximum(np.expm1(log_prediction), 0)
    return {
        "mae_usd": float(np.mean(np.abs(actual - dollar_prediction))),
        "log_rmse": float(np.sqrt(np.mean((log_actual - log_prediction) ** 2))),
    }


def pool_repeats(detail: pd.DataFrame) -> pd.DataFrame:
    """Pool held-out errors across folds before computing repeat-level RMSE."""
    rows = []
    for (group, repeat), selected in detail.groupby(["group", "repeat"], sort=False):
        n = selected["n_test"].to_numpy()
        row = {"group": group, "repeat": int(repeat), "n": int(n.sum())}
        for prefix in ["baseline", "permuted"]:
            row[f"{prefix}_mae_usd"] = float(np.average(selected[f"{prefix}_mae_usd"], weights=n))
            row[f"{prefix}_log_rmse"] = float(np.sqrt(np.average(
                selected[f"{prefix}_log_rmse"] ** 2, weights=n)))
        row["delta_mae_usd"] = row["permuted_mae_usd"] - row["baseline_mae_usd"]
        row["delta_log_rmse"] = row["permuted_log_rmse"] - row["baseline_log_rmse"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_driver_analysis(root: Path, repeats: int = 5) -> pd.DataFrame:
    """Refit the existing full model on its five fixed spatial folds only.

    Each group is shuffled jointly in held-out rows, never in training rows.
    Identical repeat permutations across groups reduce shuffle-noise differences.
    Aggregate ranges show sensitivity to shuffles, not confidence intervals.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    root = Path(root)
    tables = root / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / "data" / "processed" / "listings.csv", dtype={"id": str, "GEOID": str})
    frame = frame.reset_index(drop=True)
    for column in CATEGORICAL:
        frame[column] = frame[column].fillna("Unknown").astype(object)
    if not np.allclose(frame["log_price"], np.log1p(frame["price"]), atol=1e-12):
        raise ValueError("Processed target is inconsistent with log1p(price)")
    numeric = BASE_NUMERIC + CONTEXT_NUMERIC + TRANSIT_NUMERIC
    grouped_features = [column for columns in FEATURE_GROUPS.values() for column in columns]
    if sorted(grouped_features) != sorted(numeric + CATEGORICAL):
        raise ValueError("Feature groups must partition the exact full-model feature set")
    target = frame["log_price"].to_numpy()
    actual = frame["price"].to_numpy()
    detail, baseline_rows = [], []
    for fold, (train, test) in enumerate(GroupKFold(5).split(frame, target, groups=frame["spatial_block"]), start=1):
        print(f"Driver analysis, spatial fold {fold}/5", flush=True)
        if not set(frame.iloc[train]["spatial_block"]).isdisjoint(frame.iloc[test]["spatial_block"]):
            raise AssertionError("Spatial block leakage")
        model = feature_pipeline(numeric, CATEGORICAL, nonlinear=True)
        model.fit(frame.iloc[train], target[train])
        heldout = frame.iloc[test]
        baseline = _errors(actual[test], target[test], model.predict(heldout))
        baseline_rows.append({"fold": fold, "n_test": len(test), **baseline})
        for repeat in range(1, repeats + 1):
            seed = SEED + 1009 * fold + 17 * repeat
            for group, columns in FEATURE_GROUPS.items():
                shuffled = jointly_permute(heldout, columns, seed)
                permuted = _errors(actual[test], target[test], model.predict(shuffled))
                detail.append({
                    "group": group, "columns": ";".join(columns), "fold": fold,
                    "repeat": repeat, "seed": seed, "n_train": len(train), "n_test": len(test),
                    "baseline_mae_usd": baseline["mae_usd"],
                    "baseline_log_rmse": baseline["log_rmse"],
                    "permuted_mae_usd": permuted["mae_usd"],
                    "permuted_log_rmse": permuted["log_rmse"],
                    "delta_mae_usd": permuted["mae_usd"] - baseline["mae_usd"],
                    "delta_log_rmse": permuted["log_rmse"] - baseline["log_rmse"],
                })
    folds = pd.DataFrame(baseline_rows)
    baseline_mae = float(np.average(folds["mae_usd"], weights=folds["n_test"]))
    validation = {"candidate": "Full with transit", "validation": "Spatial 2 km 5-fold",
                  "n": len(frame), "repeats": repeats, "baseline_mae_usd": baseline_mae}
    expected_path = tables / "model_metrics.csv"
    if expected_path.exists():
        metrics = pd.read_csv(expected_path)
        expected = metrics[(metrics["validation"] == validation["validation"])
                           & (metrics["model"] == validation["candidate"])]
        if len(expected) != 1:
            raise ValueError("Expected exactly one existing full-model spatial benchmark")
        expected_mae = float(expected.iloc[0]["mae_usd"])
        validation["existing_mae_usd"] = expected_mae
        validation["baseline_matches_existing_benchmark"] = bool(
            len(frame) == int(expected.iloc[0]["n"]) and np.isclose(baseline_mae, expected_mae, rtol=1e-10, atol=1e-8))
        if not validation["baseline_matches_existing_benchmark"]:
            raise ValueError(f"Driver baseline {baseline_mae} does not match existing benchmark {expected_mae}")
    detailed = pd.DataFrame(detail)
    pooled = pool_repeats(detailed)
    rows = []
    for group, selected in pooled.groupby("group", sort=False):
        row = {"group": group, "columns": ";".join(FEATURE_GROUPS[group]),
               "n": len(frame), "folds": 5, "repeats": repeats,
               "baseline_mae_usd": float(selected["baseline_mae_usd"].iloc[0]),
               "baseline_log_rmse": float(selected["baseline_log_rmse"].iloc[0])}
        for metric in ["permuted_mae_usd", "permuted_log_rmse", "delta_mae_usd", "delta_log_rmse"]:
            row[f"{metric}_median"] = float(selected[metric].median())
            row[f"{metric}_min"] = float(selected[metric].min())
            row[f"{metric}_max"] = float(selected[metric].max())
        fold_values = detailed.loc[detailed["group"] == group]
        for metric in ["delta_mae_usd", "delta_log_rmse"]:
            row[f"fold_repeat_{metric}_min"] = float(fold_values[metric].min())
            row[f"fold_repeat_{metric}_max"] = float(fold_values[metric].max())
        rows.append(row)
    aggregate = pd.DataFrame(rows).sort_values("delta_log_rmse_median", ascending=False).reset_index(drop=True)
    aggregate["rank_log_rmse"] = np.arange(1, len(aggregate) + 1)
    aggregate["rank_dollar_mae"] = aggregate["delta_mae_usd_median"].rank(ascending=False, method="min").astype(int)
    detailed.to_csv(tables / "grouped_permutation_folds.csv", index=False)
    pooled.to_csv(tables / "grouped_permutation_repeats.csv", index=False)
    aggregate.to_csv(tables / "grouped_permutation_importance.csv", index=False)
    validation["interpretation"] = [
        "Positive deltas mean held-out error increased after shuffling a feature group.",
        "Features within each group are shuffled jointly; correlation within the group is preserved.",
        "Correlated information across groups makes importance conditional on the existing feature set.",
        "Shuffled combinations can be unrealistic; results do not identify causal effects.",
        "Repeat/fold ranges measure empirical shuffle/fold stability, not statistical confidence intervals.",
        "The full model with transit is used to retain all seven candidate feature groups, even though the without-transit candidate has slightly better unpermuted MAE.",
    ]
    (root / "results" / "driver_analysis_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Grouped held-out price-model permutation analysis")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print(run_driver_analysis(root, repeats=args.repeats).to_string(index=False))


if __name__ == "__main__":
    main()
