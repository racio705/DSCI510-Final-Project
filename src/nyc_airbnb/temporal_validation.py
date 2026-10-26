"""Protocol-locked retrospective temporal holdout for a quote benchmark."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .data import SOURCE_SPECS, _download, sha256_file, fetch_sources
from .modeling import BASE_NUMERIC, CONTEXT_NUMERIC, TRANSIT_NUMERIC, CATEGORICAL, feature_pipeline
from .temporal_data import prepare_snapshot
from .temporal_baselines import ComparableBaselines
from .temporal_cohorts import read_raw_identity_sets, strict_novel_mask, select_host_uniform

PROTOCOL_COMMIT = "b75e5a69daf625312ca2a4c582d3ddbf36657447"
PRIMARY = "Host-balanced HGB without transit"
WITH_TRANSIT = "Host-balanced HGB with transit"
ORIGINAL = "Listing-weighted HGB without transit"
BASELINES = ["comparable_listing_median", "comparable_host_median"]
FUTURE_SOURCES = {
    "2026-06-14": {"url": "https://data.insideairbnb.com/united-states/ny/new-york-city/2026-06-14/data/listings.csv.gz",
                   "sha256": "10eb56b21ec71a838b30ef47b836f3d375269baeed5ff9e5c35ff3c6a98d3ca7"},
    "2026-09-14": {"url": "https://data.insideairbnb.com/united-states/ny/new-york-city/2026-09-14/data/listings.csv.gz",
                   "sha256": "a67bf574e47aa6eba5b2262987dbd9684f206d5cd0183470056f9c936b8e7ae1"},
}


def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(path: Path, value):
    path.write_text(json.dumps(_clean_json(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ensure_sources(root: Path, offline=False) -> dict[str, Path]:
    raw = root / "data/raw/temporal"
    raw.mkdir(parents=True, exist_ok=True)
    paths, manifest = {}, {"protocol_commit": PROTOCOL_COMMIT, "sources": {}}
    for date, source in FUTURE_SOURCES.items():
        path = raw / f"listings_{date}.csv.gz"
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"Offline temporal source missing: {path}")
            _download(source["url"], path, expected_sha256=source["sha256"])
        digest = sha256_file(path)
        if digest != source["sha256"]:
            raise ValueError(f"Temporal input hash mismatch: {date}")
        paths[date] = path
        manifest["sources"][date] = {**source, "file": str(path.relative_to(root)).replace("\\", "/"),
                                     "bytes": path.stat().st_size, "sha256_verified": True}
    save_json(root / "results/temporal_source_manifest.json", manifest)
    return paths


def conformal_radius(actual, point_prediction, nominal=.8):
    residuals = np.sort(np.abs(np.log1p(actual) - np.log1p(point_prediction)))
    if not len(residuals):
        return np.nan
    rank = min(int(np.ceil((len(residuals) + 1) * nominal)), len(residuals))
    return float(residuals[rank - 1])


def prediction_bounds(point, radius):
    point = np.asarray(point)
    return np.maximum(np.expm1(np.log1p(point) - radius), 0), np.expm1(np.log1p(point) + radius)


def quote_metrics(actual, point, radius):
    actual, point = np.asarray(actual), np.asarray(point)
    if not len(actual):
        return {"n": 0}
    lower, upper = prediction_bounds(point, radius)
    absolute = np.abs(actual - point)
    return {"n": len(actual), "mae_usd": float(absolute.mean()),
            "rmse_usd": float(np.sqrt(np.mean(absolute ** 2))),
            "median_ae_usd": float(np.median(absolute)),
            "median_ape_pct": float(np.median(absolute / actual) * 100),
            "within_25pct": float(np.mean(absolute / actual <= .25) * 100),
            "r2_log": float(r2_score(np.log1p(actual), np.log1p(point))) if len(actual) > 1 else np.nan,
            "coverage_80_pct": float(np.mean((actual >= lower) & (actual <= upper)) * 100),
            "median_interval_width_usd": float(np.median(upper - lower)),
            "median_relative_width": float(np.median((upper - lower) / (point + 1)))}


def paired_intervals(actual, predictions, bootstrap=3000):
    """Same paired host resamples for the three predeclared contrasts."""
    actual = np.asarray(actual)
    if len(actual) < 2:
        return pd.DataFrame(columns=["contrast", "n_hosts", "estimate_pct", "family_ci_lower_pct",
                                     "family_ci_upper_pct", "individual_interval_level_pct", "bootstrap_replicates"])
    losses = {name: np.abs(actual - np.asarray(point)) for name, point in predictions.items()}
    rng = np.random.default_rng(42)
    draws = {name: [] for name in [*BASELINES, "transit_relative_mae"]}
    for _ in range(bootstrap):
        index = rng.integers(0, len(actual), size=len(actual))
        primary_mae = losses[PRIMARY][index].mean()
        for baseline in BASELINES:
            baseline_mae = losses[baseline][index].mean()
            draws[baseline].append(100 * (1 - primary_mae / baseline_mae))
        draws["transit_relative_mae"].append(100 * (losses[WITH_TRANSIT][index].mean() / primary_mae - 1))
    alpha = .05 / 3
    rows = []
    for name, values in draws.items():
        estimate = (100 * (1 - losses[PRIMARY].mean() / losses[name].mean()) if name in BASELINES
                    else 100 * (losses[WITH_TRANSIT].mean() / losses[PRIMARY].mean() - 1))
        rows.append({"contrast": name, "n_hosts": len(actual), "estimate_pct": estimate,
                     "family_ci_lower_pct": float(np.quantile(values, alpha / 2)),
                     "family_ci_upper_pct": float(np.quantile(values, 1 - alpha / 2)),
                     "individual_interval_level_pct": 100 * (1 - alpha), "bootstrap_replicates": bootstrap})
    return pd.DataFrame(rows)


def run_temporal_validation(root: Path, offline=False) -> dict:
    root = Path(root)
    tables = root / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    fetch_sources(root, offline=offline)
    source_paths = ensure_sources(root, offline=offline)
    old_path = root / "data/raw" / SOURCE_SPECS["airbnb"]["filename"]
    if sha256_file(old_path) != SOURCE_SPECS["airbnb"]["sha256"]:
        raise ValueError("October training source hash mismatch")
    train, train_hosts, train_quality = prepare_snapshot(root, old_path)
    if train_hosts.isna().any() or train_hosts.astype(str).str.strip().eq("").any():
        raise ValueError("Training requires a known host for every eligible October row")
    x_train = train.drop(columns=["price", "log_price"])
    for column in CATEGORICAL:
        x_train[column] = x_train[column].fillna("Unknown").astype(object)
    y_train = np.log1p(train.price.to_numpy())
    counts = train_hosts.map(train_hosts.value_counts()).to_numpy()
    weight = 1 / counts
    weight /= weight.mean()
    baselines = ComparableBaselines().fit(x_train.assign(price=train.price.to_numpy()), train_hosts)
    models = {}
    for name, transit, balanced in [(PRIMARY, False, True), (WITH_TRANSIT, True, True), (ORIGINAL, False, False)]:
        numeric = BASE_NUMERIC + CONTEXT_NUMERIC + (TRANSIT_NUMERIC if transit else [])
        model = feature_pipeline(numeric, CATEGORICAL, nonlinear=True)
        kwargs = {"model__sample_weight": weight} if balanced else {}
        model.fit(x_train, y_train, **kwargs)
        models[name] = model
    # No June/September listing labels have been used to fit the point models.
    trained_at = datetime.now(timezone.utc).isoformat()
    identity_old, host_old = read_raw_identity_sets(old_path)
    frames, host_arrays, qualities = {"2025-10-01": train}, {"2025-10-01": train_hosts}, {"2025-10-01": train_quality}
    points, levels, prediction_hashes = {}, {}, {}
    for date, path in source_paths.items():
        frame, hosts, quality = prepare_snapshot(root, path)
        frames[date], host_arrays[date], qualities[date] = frame, hosts, quality
        x_future = frame.drop(columns=["price", "log_price"])
        for column in CATEGORICAL:
            x_future[column] = x_future[column].fillna("Unknown").astype(object)
        comparable = baselines.predict(x_future)
        point = {name: np.maximum(np.expm1(model.predict(x_future)), 0) for name, model in models.items()}
        point.update({name: comparable[name].to_numpy() for name in BASELINES})
        points[date], levels[date] = point, comparable["comparable_level"].to_numpy()
        destination = processed / f"frozen_temporal_predictions_{date}.csv"
        pd.DataFrame({"id": frame.id, **point, "comparable_level": levels[date]}).to_csv(destination, index=False)
        prediction_hashes[date] = sha256_file(destination)
        frame.to_csv(processed / f"temporal_features_{date}.csv", index=False)
    # Both future point-prediction artifacts are frozen before future outcome scoring.
    save_json(root / "results/temporal_prediction_freeze.json", {
        "protocol_commit": PROTOCOL_COMMIT, "point_models_trained_at_utc": trained_at,
        "future_prediction_sha256": prediction_hashes, "target_columns_passed_to_prediction": False,
        "phase": "point predictions frozen before June calibration / September scoring"})

    june, september = frames["2026-06-14"], frames["2026-09-14"]
    june_hosts, september_hosts = host_arrays["2026-06-14"], host_arrays["2026-09-14"]
    strict_june = strict_novel_mask(june, june_hosts, identity_old, host_old)
    cal_positions = select_host_uniform(june.iloc[np.flatnonzero(strict_june)].reset_index(drop=True),
                                        june_hosts.iloc[np.flatnonzero(strict_june)].reset_index(drop=True))
    calibration_index = np.flatnonzero(strict_june)[cal_positions]
    identity_june, host_june = read_raw_identity_sets(source_paths["2026-06-14"])
    prior_ids, prior_hosts = identity_old | identity_june, host_old | host_june
    strict_final = strict_novel_mask(september, september_hosts, prior_ids, prior_hosts)
    strict_index = np.flatnonzero(strict_final)
    primary_positions = select_host_uniform(september.iloc[strict_index].reset_index(drop=True),
                                            september_hosts.iloc[strict_index].reset_index(drop=True))
    primary_index = strict_index[primary_positions]
    if september_hosts.iloc[primary_index].nunique() != len(primary_index):
        raise AssertionError("Primary test must contain exactly one listing per host")
    radii = {name: conformal_radius(june.price.to_numpy()[calibration_index], point[calibration_index])
             for name, point in points["2026-06-14"].items()}
    cohorts = {
        "All eligible September listings": np.arange(len(september)),
        "September listings absent from all prior raw IDs": np.flatnonzero(~september.id.isin(prior_ids)),
        "All strict novel-host and novel-listing rows": strict_index,
        "PRIMARY one listing per strict novel host": primary_index,
    }
    rows, subgroup_rows, level_rows = [], [], []
    for cohort, index in cohorts.items():
        actual = september.price.to_numpy()[index]
        for name, point in points["2026-09-14"].items():
            rows.append({"cohort": cohort, "model": name, "n_hosts": september_hosts.iloc[index].nunique(),
                         **quote_metrics(actual, point[index], radii[name])})
        fallback = pd.Series(levels["2026-09-14"][index]).value_counts().sort_index()
        level_rows.extend({"cohort": cohort, "comparable_level": level, "n": int(n)} for level, n in fallback.items())
    subset = september.iloc[primary_index].copy()
    subset["_position"] = primary_index
    for (borough, room), selected in subset.groupby(["borough", "room_type"], observed=True):
        index = selected._position.to_numpy()
        for name in [PRIMARY, *BASELINES]:
            subgroup_rows.append({"borough": borough, "room_type": room, "model": name,
                                  "small_sample_n_lt30": len(index) < 30,
                                  **quote_metrics(september.price.to_numpy()[index], points["2026-09-14"][name][index], radii[name])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(tables / "temporal_holdout_metrics.csv", index=False)
    subgroup_frame = pd.DataFrame(subgroup_rows) if subgroup_rows else pd.DataFrame(
        columns=["borough", "room_type", "model", "small_sample_n_lt30", "n", "mae_usd"])
    subgroup_frame.to_csv(tables / "temporal_holdout_subgroups.csv", index=False)
    pd.DataFrame(level_rows).to_csv(tables / "temporal_baseline_fallback.csv", index=False)
    primary_points = {name: point[primary_index] for name, point in points["2026-09-14"].items()}
    contrasts = paired_intervals(september.price.to_numpy()[primary_index], primary_points)
    contrasts.to_csv(tables / "temporal_paired_intervals.csv", index=False)
    main = metrics[(metrics.cohort == "PRIMARY one listing per strict novel host") & (metrics.model == PRIMARY)].iloc[0]
    enough = len(calibration_index) >= 300 and len(primary_index) >= 300
    superiority = {}
    for baseline in BASELINES:
        selected = contrasts[contrasts.contrast == baseline]
        superiority[baseline] = bool(len(selected) and selected.iloc[0].estimate_pct >= 10 and selected.iloc[0].family_ci_lower_pct > 0)
    gates = {"calibration_and_test_at_least_300_hosts": enough,
             "beats_listing_comparables_by_10pct_with_positive_interval": superiority[BASELINES[0]],
             "beats_host_comparables_by_10pct_with_positive_interval": superiority[BASELINES[1]],
             "median_ape_at_most_30pct": bool(main.get("median_ape_pct", np.inf) <= 30),
             "coverage_80_at_least_75pct": bool(main.get("coverage_80_pct", 0) >= 75),
             "median_relative_interval_width_at_most_1_5": bool(main.get("median_relative_width", np.inf) <= 1.5)}
    equivalence = contrasts[contrasts.contrast == "transit_relative_mae"]
    equivalent = bool(enough and len(equivalence) and equivalence.iloc[0].family_ci_lower_pct >= -2 and equivalence.iloc[0].family_ci_upper_pct <= 2)
    decision = "met_predeclared_offline_criteria" if all(gates.values()) else "criteria_not_met" if enough else "insufficient_independent_sample"
    from .temporal_replication import evaluate_market_replication
    replication = evaluate_market_replication({date: (frame, host_arrays[date]) for date, frame in frames.items()}, root)
    summary = {"protocol_commit": PROTOCOL_COMMIT, "primary_model": PRIMARY,
               "decision": decision, "gates": gates, "transit_practically_equivalent_within_2pct": equivalent,
               "calibration_hosts": len(calibration_index), "primary_test_hosts": len(primary_index),
               "training_listings": len(train), "training_hosts": train_hosts.nunique(),
               "quantile_radii": radii, "metrics": rows, "paired_intervals": contrasts.to_dict(orient="records"),
               "market_replication": replication, "snapshot_quality": qualities,
               "prediction_freeze_sha256": prediction_hashes,
               "scope": "Retrospective temporal transfer with contemporaneous attributes; historical asking prices, not achieved rates or future bookings"}
    save_json(root / "results/temporal_validation.json", summary)
    model_dir = root / "data/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "baselines": baselines, "radii": radii,
                 "protocol_commit": PROTOCOL_COMMIT, "decision": decision}, model_dir / "temporal_quote_benchmark.joblib")
    save_json(root / "results/temporal_model_card.json", {
        "name": "NYC long-stay asking-price screening benchmark", "protocol_commit": PROTOCOL_COMMIT,
        "primary_model": PRIMARY, "decision": decision, "gates": gates,
        "training_snapshot": "2025-10-01", "calibration_snapshot": "2026-06-14", "test_snapshot": "2026-09-14",
        "training_listings": len(train), "training_hosts": train_hosts.nunique(),
        "calibration_hosts": len(calibration_index), "strict_final_hosts": len(primary_index),
        "primary_metrics": main.to_dict(),
        "scope": summary["scope"],
        "intended_use": "Long-stay non-hotel asking-price comparison and screening, with individual uncertainty",
        "limitations": ["No booked-price, demand, occupancy, revenue or causal claim", "Temporal coverage is empirical, not guaranteed",
                        "Queens private-room precision is below the stated target; several subgroups are small",
                        "See every subgroup and all predeclared gates in the validation report"],
        "artifacts": {"local_model": "data/models/temporal_quote_benchmark.joblib", "metrics": "results/temporal_validation.json",
                      "source_hashes": "results/temporal_source_manifest.json", "prediction_freeze": "results/temporal_prediction_freeze.json"},
    })
    from .temporal_reporting import render_temporal_report
    render_temporal_report(root)
    print("Temporal decision:", decision, "calibration hosts:", len(calibration_index), "final hosts:", len(primary_index), flush=True)
    print(metrics[metrics.cohort == "PRIMARY one listing per strict novel host"].to_string(index=False), flush=True)
    print(contrasts.to_string(index=False), flush=True)
    return summary
