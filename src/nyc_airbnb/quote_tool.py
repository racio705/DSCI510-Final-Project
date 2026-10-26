"""Local historical asking-price comparison from the frozen evaluation bundle.

Inputs contain property information only. Geography is inferred from the frozen
reference data; no target price is accepted or fabricated during preparation.
The saved adoption decision and nominal-interval limitations remain visible.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import json
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd

from .data import accessibility_features, attach_tracts
from .modeling import BASE_NUMERIC, CONTEXT_NUMERIC, TRANSIT_NUMERIC, CATEGORICAL
from .temporal_baselines import KEY_LEVELS, normalize_comparable_keys
from .temporal_data import _frozen_references

PROTOCOL_COMMIT = "b75e5a69daf625312ca2a4c582d3ddbf36657447"
PRIMARY = "Host-balanced HGB without transit"
BASELINE_NAMES = ["comparable_listing_median", "comparable_host_median"]
MODEL_RELATIVE_PATH = Path("data/models/temporal_quote_benchmark.joblib")
REQUIRED = {"latitude", "longitude", "room_type", "property_type", "accommodates",
            "bedrooms", "bathrooms", "minimum_nights"}
OPTIONAL = {"beds", "number_of_reviews", "review_scores_rating", "calculated_host_listings_count",
            "neighbourhood_cleansed", "neighborhood", "hypothetical", "scenario_note"}
TARGET_FIELDS = {"price", "log_price", "target_price", "actual_price", "nightly_price", "quoted_price"}
NUMERIC_INPUTS = ["accommodates", "bedrooms", "bathrooms", "minimum_nights", "beds",
                  "number_of_reviews", "review_scores_rating", "calculated_host_listings_count"]
COUNT_FIELDS = {"accommodates", "bedrooms", "minimum_nights", "beds",
                "number_of_reviews", "calculated_host_listings_count"}


def _missing(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip()) or bool(pd.isna(value))


def _number(value, name: str, *, optional=False) -> float:
    if _missing(value):
        if optional:
            return np.nan
        raise ValueError(f"{name} is required and must be numeric")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_request(request: dict) -> dict:
    """Validate recorded features independently of a target or model output."""
    if not isinstance(request, dict):
        raise ValueError("Each quote request must be a JSON object or a CSV row")
    if any(not isinstance(key, str) for key in request):
        raise ValueError("Request field names must be strings")
    targets = {key for key in request if key.casefold() in TARGET_FIELDS}
    if targets:
        raise ValueError(f"Target-price fields are not accepted: {sorted(targets)}")
    unknown = set(request).difference(REQUIRED | OPTIONAL)
    if unknown:
        raise ValueError(f"Unsupported fields: {sorted(unknown)}")
    missing = REQUIRED.difference(request)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    result = dict(request)
    result["latitude"] = _number(request["latitude"], "latitude")
    result["longitude"] = _number(request["longitude"], "longitude")
    if not -90 <= result["latitude"] <= 90 or not -180 <= result["longitude"] <= 180:
        raise ValueError("Latitude/longitude are outside valid WGS84 bounds")
    room_lookup = {value.casefold(): value for value in ["Entire home/apt", "Private room", "Shared room"]}
    room = str(request["room_type"]).strip()
    if room.casefold() not in room_lookup:
        raise ValueError("room_type must be Entire home/apt, Private room or Shared room; hotels are outside scope")
    result["room_type"] = room_lookup[room.casefold()]
    if _missing(request["property_type"]) or not isinstance(request["property_type"], str):
        raise ValueError("property_type must be a nonempty recorded property category")
    result["property_type"] = request["property_type"].strip()
    if any(term in result["property_type"].casefold() for term in ["hotel", "hostel", "resort"]):
        raise ValueError("Hotel, hostel and resort properties are outside the evaluated scope")
    for name in NUMERIC_INPUTS:
        result[name] = _number(request.get(name), name, optional=name not in REQUIRED)
        value = result[name]
        if np.isnan(value):
            continue
        if value < 0 or (name in COUNT_FIELDS and value != np.floor(value)):
            raise ValueError(f"{name} must be a nonnegative {'integer' if name in COUNT_FIELDS else 'number'}")
    if result["accommodates"] < 1:
        raise ValueError("accommodates must be at least one")
    if result["minimum_nights"] < 30:
        raise ValueError("This benchmark requires advertised minimum_nights >=30")
    if np.isfinite(result["review_scores_rating"]) and not 0 <= result["review_scores_rating"] <= 5:
        raise ValueError("review_scores_rating must use the 0-to-5 scale")
    neighborhood = request.get("neighbourhood_cleansed", request.get("neighborhood"))
    if ("neighbourhood_cleansed" in request and "neighborhood" in request
            and str(request["neighbourhood_cleansed"]).strip() != str(request["neighborhood"]).strip()):
        raise ValueError("Conflicting neighborhood fields")
    if not _missing(neighborhood) and not isinstance(neighborhood, str):
        raise ValueError("Neighborhood must be a recorded string category")
    result["neighbourhood_cleansed"] = "Unknown" if _missing(neighborhood) else neighborhood.strip()
    hypothetical = request.get("hypothetical", False)
    if isinstance(hypothetical, str) and hypothetical.strip().casefold() in {"true", "false"}:
        hypothetical = hypothetical.strip().casefold() == "true"
    if _missing(hypothetical):
        hypothetical = False
    if not isinstance(hypothetical, (bool, np.bool_)):
        raise ValueError("hypothetical must be true or false")
    result["hypothetical"] = bool(hypothetical)
    note = request.get("scenario_note")
    if not _missing(note) and not isinstance(note, str):
        raise ValueError("scenario_note must be a string")
    result["scenario_note"] = None if _missing(note) else note.strip()
    return result


def build_quote_features(root: Path, request: dict, *, references=None) -> tuple[pd.DataFrame, dict]:
    """Enrich one feature-only request; infer borough rather than trusting input."""
    request = validate_request(request)
    tracts, stations, _ = _frozen_references(Path(root)) if references is None else references
    row = {name: request[name] for name in NUMERIC_INPUTS}
    row.update({name: request[name] for name in ["latitude", "longitude", "room_type", "property_type", "neighbourhood_cleansed"]})
    points = gpd.GeoDataFrame([row], geometry=gpd.points_from_xy(
        [request["longitude"]], [request["latitude"]]), crs="EPSG:4326")
    joined, _ = attach_tracts(points, tracts.drop(columns="COUNTYFP", errors="ignore"))
    if len(joined) != 1 or joined["GEOID"].isna().any():
        raise ValueError("Coordinates do not match one NYC census tract; no benchmark can be generated")
    accessibility = accessibility_features(joined, stations)
    for name in accessibility:
        joined[name] = accessibility[name]
    joined["log_minimum_nights"] = np.log1p(joined["minimum_nights"])
    joined["log_review_count"] = np.log1p(joined["number_of_reviews"])
    joined["log_host_listings_count"] = np.log1p(joined["calculated_host_listings_count"])
    features = pd.DataFrame(joined.drop(columns="geometry")).reset_index(drop=True)
    if {"price", "log_price"}.intersection(features):
        raise AssertionError("Target prices must not enter quote feature preparation")
    required_features = BASE_NUMERIC + CONTEXT_NUMERIC + TRANSIT_NUMERIC + CATEGORICAL
    if set(required_features).difference(features):
        raise ValueError("Frozen geographic references do not provide the required model features")
    for name in CATEGORICAL:
        features[name] = features[name].astype(object)
    metadata = {"hypothetical": request["hypothetical"], "scenario_note": request["scenario_note"],
                "neighborhood_was_supplied": request["neighbourhood_cleansed"] != "Unknown",
                "optional_inputs_missing": [name for name in ["beds", "number_of_reviews", "review_scores_rating", "calculated_host_listings_count"]
                                            if not np.isfinite(request[name])]}
    return features, metadata


def load_bundle(root: Path) -> dict:
    """Load only the locally generated, protocol-versioned benchmark bundle."""
    path = Path(root) / MODEL_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError("Frozen model bundle missing. First run python src/validate_temporal.py --offline")
    bundle = joblib.load(path)
    if not {"models", "baselines", "radii", "protocol_commit", "decision"}.issubset(bundle):
        raise ValueError("Incomplete frozen benchmark bundle")
    if bundle["protocol_commit"] != PROTOCOL_COMMIT:
        raise ValueError("Benchmark bundle belongs to a different temporal evaluation protocol")
    if PRIMARY not in bundle["models"]:
        raise ValueError("Frozen bundle does not contain the locked primary model")
    return bundle


def _estimate(point: float, radius) -> dict:
    point = float(point)
    if not np.isfinite(point) or point < 0:
        raise ValueError("Frozen benchmark produced an invalid point estimate")
    radius = float(radius) if radius is not None else np.nan
    interval = None
    if np.isfinite(radius) and radius >= 0:
        lower = max(float(np.expm1(np.log1p(point) - radius)), 0)
        upper = float(np.expm1(np.log1p(point) + radius))
        interval = {"lower_usd": round(lower, 2), "upper_usd": round(upper, 2), "nominal_level_pct": 80}
    return {"point_usd_per_night": round(point, 2), "nominal_prediction_interval_80pct": interval}


def _comparable_support(baselines, features: pd.DataFrame, level: int) -> dict:
    if not 1 <= level <= 9:
        raise ValueError("Unexpected comparable-price fallback level")
    dimensions = list(KEY_LEVELS[level - 1]) if level <= len(KEY_LEVELS) else []
    support = {"level": level, "matched_dimensions": dimensions or ["global"],
               "training_listings": None, "training_hosts": None}
    if level == 9:
        support.update(training_listings=int(baselines.training_n_), training_hosts=int(baselines.training_hosts_))
    elif hasattr(baselines, "support_"):
        table = baselines.support_[level - 1]
        key = normalize_comparable_keys(features).iloc[0]
        mask = np.ones(len(table), dtype=bool)
        for column in dimensions:
            mask &= table[column].eq(key[column]).to_numpy()
        selected = table.loc[mask]
        if len(selected) == 1:
            support.update(training_listings=int(selected.iloc[0].n_listings), training_hosts=int(selected.iloc[0].n_hosts))
    return support


def subgroup_diagnostic(table: pd.DataFrame, borough: str, room_type: str) -> dict:
    """Expose observed subgroup weaknesses without changing the frozen decision."""
    if table.empty:
        return {"status": "not_available", "flags": []}
    selected = table[(table["model"] == PRIMARY) & (table["borough"] == borough) & (table["room_type"] == room_type)]
    if len(selected) != 1:
        return {"status": "not_available", "flags": []}
    row = selected.iloc[0]
    flags = []
    if int(row.n) < 30:
        flags.append("Fewer than 30 matching holdout examples; subgroup evidence is limited.")
    if row.median_ape_pct > 30:
        flags.append("Matching subgroup median percentage error exceeded the 30% precision target.")
    if row.coverage_80_pct < 75:
        flags.append("Matching subgroup empirical interval coverage was below 75%.")
    return {"status": "observed_holdout_subgroup", "n": int(row.n),
            "mae_usd": float(row.mae_usd), "median_ape_pct": float(row.median_ape_pct),
            "coverage_80_pct": float(row.coverage_80_pct), "flags": flags,
            "note": "Descriptive subgroup diagnostics, not a conditional coverage guarantee or a new adoption gate."}


def quote_requests(root: Path, requests: list[dict], *, bundle=None, references=None) -> list[dict]:
    """Return historical comparisons without changing any fitted model or radius."""
    validated = [validate_request(request) for request in requests]
    if not validated:
        raise ValueError("At least one quote request is required")
    bundle = load_bundle(root) if bundle is None else bundle
    if bundle["protocol_commit"] != PROTOCOL_COMMIT or PRIMARY not in bundle["models"]:
        raise ValueError("Expected the frozen primary model and evaluation protocol")
    references = _frozen_references(Path(root)) if references is None else references
    subgroup_path = Path(root) / "results/tables/temporal_holdout_subgroups.csv"
    subgroup_table = pd.read_csv(subgroup_path) if subgroup_path.exists() else pd.DataFrame()
    outputs = []
    for request in validated:
        features, metadata = build_quote_features(root, request, references=references)
        point = max(float(np.expm1(bundle["models"][PRIMARY].predict(features)[0])), 0)
        comparable = bundle["baselines"].predict(features).iloc[0]
        estimates = {PRIMARY: _estimate(point, bundle["radii"].get(PRIMARY))}
        estimates.update({name: _estimate(comparable[name], bundle["radii"].get(name)) for name in BASELINE_NAMES})
        level = int(comparable["comparable_level"])
        fallback = _comparable_support(bundle["baselines"], features, level)
        warnings = []
        if not metadata["neighborhood_was_supplied"]:
            warnings.append("Recorded neighborhood was not supplied; comparable medians use a broader supported fallback cell.")
        else:
            warnings.append("The supplied neighborhood category is not verified against neighborhood boundary data.")
        if metadata["optional_inputs_missing"]:
            warnings.append("Missing optional model features use the frozen training-fitted imputation.")
        if level == 9:
            warnings.append("Comparable medians fall back to the global training sample rather than a local product cell.")
        if estimates[PRIMARY]["nominal_prediction_interval_80pct"] is None:
            warnings.append("A finite calibration radius is unavailable; no calibrated interval is shown.")
        decision = str(bundle["decision"])
        if decision == "met_predeclared_offline_criteria":
            decision_note = "The locked primary model met researcher-chosen offline screening criteria. This is not automatic-pricing approval or demonstrated business impact."
        else:
            decision_note = "The locked primary model did not establish the predeclared adoption criteria; retain this as an experimental historical reference."
        place = features.iloc[0]
        diagnostic = subgroup_diagnostic(subgroup_table, str(place.borough), str(place.room_type))
        warnings.extend(diagnostic["flags"])
        input_features = {name: float(request[name]) if np.isfinite(request[name]) else None
                          for name in ["latitude", "longitude", *NUMERIC_INPUTS]}
        input_features.update({name: request[name] for name in ["room_type", "property_type", "neighbourhood_cleansed"]})
        outputs.append({
            "benchmark_type": "historical_advertised_nightly_price", "currency": "USD",
            **metadata, "input_features": input_features, "primary_model": PRIMARY,
            "point_quote_usd_per_night": estimates[PRIMARY]["point_usd_per_night"],
            "estimates": estimates, "comparable_fallback": fallback,
            "location_context": {"borough": str(place.borough), "census_tract": str(place.GEOID),
                                 "recorded_neighborhood": str(place.neighbourhood_cleansed),
                                 "nearest_station": str(place.nearest_station),
                                 "straight_line_subway_distance_m": round(float(place.dist_km) * 1000, 1)},
            "adoption_decision": decision, "adoption_decision_note": decision_note,
            "observed_subgroup_validation": diagnostic,
            "protocol_commit": bundle["protocol_commit"],
            "training_snapshot": "2025-10-01", "calibration_snapshot": "2026-06-14", "holdout_snapshot": "2026-09-14",
            "interval_note": "Nominal 80% log-residual calibration interval. Temporal drift removes an exchangeability guarantee; this is not guaranteed 80% coverage for this request.",
            "scope_note": "Historical asking-price comparison for non-hotel NYC listings with advertised minimum stay >=30 nights. Not achieved booking rates, valuation, revenue or an offer to book.",
            "warnings": warnings,
        })
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Compare a long-stay NYC listing with frozen historical asking-price benchmarks")
    parser.add_argument("--input", type=Path, required=True, help="JSON object/list or a CSV of property features, with no target price")
    parser.add_argument("--output", type=Path, help="Optional JSON output file")
    args = parser.parse_args()
    try:
        if args.input.suffix.casefold() == ".csv":
            records = pd.read_csv(args.input).to_dict(orient="records")
            single = False
        else:
            content = json.loads(args.input.read_text(encoding="utf-8-sig"))
            single = isinstance(content, dict)
            records = [content] if single else content
            if not isinstance(records, list):
                raise ValueError("JSON input must be an object or a list of objects")
        results = quote_requests(Path(__file__).resolve().parents[2], records)
        serialized = json.dumps(results[0] if single else results, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(args.output.name + ".tmp")
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(args.output)
        print(serialized, end="")
    except (ValueError, TypeError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"Quote request rejected: {exc}\n")


if __name__ == "__main__":
    main()
