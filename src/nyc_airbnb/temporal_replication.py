"""Evaluate the market benchmark using the frozen temporal protocol.

This module never reads listing files or discovers additional cohorts. The
caller supplies already prepared, eligible long-stay frames and aligned host
identifiers, after its prediction-freeze step. Only aggregate results leave
this module; host identities and listing-level prices are never exported.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOL_COMMIT = "b75e5a69daf625312ca2a4c582d3ddbf36657447"
CONTEXT_PERIOD = "2025-10-01"
FUTURE_PERIODS = ("2026-06-14", "2026-09-14")
MIN_LISTINGS_PER_BOROUGH = 100
MIN_HOSTS_PER_BOROUGH = 30
MIN_LOWER_PRICE_PCT = 10.0

PRODUCTS = {
    "entire_1br_2guests": ("Entire home/apt", 2, 2),
    "entire_1br_2to4guests": ("Entire home/apt", 2, 4),
    "private_1br_1to2guests": ("Private room", 1, 2),
}


def _canonical_period(value: str) -> str:
    """Accept only the three fixed snapshots, allowing a year-month label."""
    label = str(value)
    for period in (CONTEXT_PERIOD, *FUTURE_PERIODS):
        if label == period or label == period[:7]:
            return period
    raise ValueError(f"Snapshot is outside the frozen protocol: {value}")


def _validated_frame(frame: pd.DataFrame, hosts: pd.Series) -> pd.DataFrame:
    required = {"id", "price", "room_type", "borough", "bedrooms",
                "accommodates", "minimum_nights", "dist_km"}
    if not required.issubset(frame):
        raise ValueError(f"Replication input missing columns: {sorted(required.difference(frame))}")
    if not isinstance(hosts, pd.Series) or not frame.index.equals(hosts.index):
        raise ValueError("Host Series must have exactly the same ordered index as its feature frame")
    if frame.empty or frame["id"].isna().any() or not frame["id"].is_unique:
        raise ValueError("Replication input requires nonempty, unique, nonmissing listing IDs")
    host_values = hosts.astype("string")
    if host_values.isna().any() or host_values.str.strip().eq("").any():
        raise ValueError("Every eligible replication listing needs a nonmissing host account identifier")
    data = frame.copy()
    for column in ["price", "bedrooms", "accommodates", "minimum_nights", "dist_km"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if not np.isfinite(data["price"]).all() or not data["price"].gt(0).all():
        raise ValueError("Replication population must contain finite positive advertised prices")
    if not data["minimum_nights"].ge(30).all():
        raise ValueError("Supply the eligible long-stay population, with minimum_nights >=30")
    if not np.isfinite(data["dist_km"]).all() or not data["dist_km"].ge(0).all():
        raise ValueError("Replication population requires finite nonnegative station distances")
    hotel = data["room_type"].eq("Hotel room")
    if "property_type" in data:
        hotel |= data["property_type"].fillna("").astype(str).str.contains("hotel|hostel|resort", case=False, regex=True)
    if hotel.any():
        raise ValueError("Hospitality listings remain in the supplied replication population")
    data["_host"] = host_values
    return data


def _borough_summary(data: pd.DataFrame) -> dict:
    """Listing-weighted and host-sensitive summaries, without identity outputs.

    A tie for largest account is broken lexicographically on the account ID,
    without referring to prices. IDs are used only inside this function.
    """
    n = len(data)
    counts = data.groupby("_host", sort=True).size()
    largest_n = int(counts.max()) if len(counts) else 0
    if len(counts):
        largest_host = sorted(counts[counts.eq(largest_n)].index, key=str)[0]
        remaining = data[data["_host"].ne(largest_host)]
    else:
        remaining = data
    host_medians = data.groupby("_host", sort=True)["price"].median()
    return {
        "n": n,
        "hosts": int(len(counts)),
        "median_price_usd": float(data["price"].median()) if n else np.nan,
        "largest_host_n": largest_n,
        "largest_host_share_pct": float(100 * largest_n / n) if n else np.nan,
        "n_excluding_largest_host": len(remaining),
        "median_excluding_largest_host_usd": float(remaining["price"].median()) if len(remaining) else np.nan,
        "median_of_host_medians_usd": float(host_medians.median()) if len(host_medians) else np.nan,
    }


def _relative_gap(brooklyn: float, manhattan: float) -> float:
    if not np.isfinite(brooklyn) or not np.isfinite(manhattan) or manhattan <= 0:
        return np.nan
    return float(100 * (brooklyn / manhattan - 1))


def _period_gates(row: dict) -> dict:
    """Apply unchanged listing/host support and point-replication criteria."""
    supported = all([
        row["n_manhattan"] >= MIN_LISTINGS_PER_BOROUGH,
        row["n_brooklyn"] >= MIN_LISTINGS_PER_BOROUGH,
        row["hosts_manhattan"] >= MIN_HOSTS_PER_BOROUGH,
        row["hosts_brooklyn"] >= MIN_HOSTS_PER_BOROUGH,
    ])
    listing_gap = row["brooklyn_gap_pct"]
    exclusion_gap = row["brooklyn_gap_excluding_largest_hosts_pct"]
    gap_pass = bool(np.isfinite(listing_gap) and listing_gap <= -MIN_LOWER_PRICE_PCT)
    direction_pass = bool(np.isfinite(exclusion_gap) and exclusion_gap < 0)
    return {
        "sample_gate_met": bool(supported),
        "listing_gap_at_least_10pct_lower": gap_pass,
        "lower_after_largest_host_exclusion": direction_pass,
        "period_criteria_met": bool(supported and gap_pass and direction_pass),
    }


def _decide_main_replication(rows: list[dict]) -> dict:
    """October cannot count toward the two-period replication decision."""
    main = {row["period"]: row for row in rows if row["is_principal"] and row["period"] in FUTURE_PERIODS}
    missing = [period for period in FUTURE_PERIODS if period not in main]
    undersupported = [period for period, row in main.items() if not row["sample_gate_met"]]
    failures = [period for period, row in main.items()
                if row["sample_gate_met"] and not row["period_criteria_met"]]
    if missing or undersupported:
        status = "insufficient_sample"
    elif failures:
        status = "not_replicated"
    else:
        status = "replicated"
    return {
        "status": status,
        "recurring_benchmark_supported": status == "replicated",
        "missing_required_periods": missing,
        "periods_below_sample_gate": undersupported,
        "supported_periods_failing_price_or_exclusion_criteria": failures,
        "required_future_periods": list(FUTURE_PERIODS),
        "october_is_historical_context_only": True,
    }


def evaluate_market_replication(
        frames: dict[str, tuple[pd.DataFrame, pd.Series]], root: Path | str) -> dict:
    """Export fixed M/Brooklyn benchmark rows and the protocol's decision.

    Supply all eligible long-stay listings for each snapshot, rather than the
    host-uniform novel-host sample used for the separate prediction decision.
    The principal market comparison is entire home, one bedroom, capacity two,
    minimum exactly 30 nights and station distance <=500 m. Other product and
    800 m rows are sensitivities and never determine the principal status.
    """
    root = Path(root)
    period_frames = {}
    for label, (frame, hosts) in frames.items():
        period = _canonical_period(label)
        if period in period_frames:
            raise ValueError(f"More than one input supplied for the same protocol period: {period}")
        period_frames[period] = _validated_frame(frame, hosts)
    if not period_frames:
        raise ValueError("No eligible snapshot frames were supplied")

    rows = []
    for period in (CONTEXT_PERIOD, *FUTURE_PERIODS):
        if period not in period_frames:
            continue
        data = period_frames[period]
        for product_id, (room_type, capacity_min, capacity_max) in PRODUCTS.items():
            product = (data["room_type"].eq(room_type) & data["bedrooms"].eq(1)
                       & data["accommodates"].between(capacity_min, capacity_max)
                       & data["minimum_nights"].eq(30))
            for threshold in [500, 800]:
                selected = data[product & data["dist_km"].le(threshold / 1000)]
                principal = product_id == "entire_1br_2guests" and threshold == 500
                row = {
                    "period": period,
                    "data_role": "historical_context" if period == CONTEXT_PERIOD else
                                 "calibration_period_replication" if period == FUTURE_PERIODS[0] else
                                 "final_holdout_replication",
                    "product_id": product_id,
                    "maximum_station_distance_m": threshold,
                    "minimum_nights": 30,
                    "is_principal": principal,
                    "comparison_role": "principal" if principal else "sensitivity_only",
                    "eligible_long_stay_population_n": len(data),
                }
                summaries = {}
                for borough in ["Manhattan", "Brooklyn"]:
                    result = _borough_summary(selected[selected["borough"].eq(borough)])
                    summaries[borough] = result
                    for statistic, value in result.items():
                        # Place names are column labels; IDs never enter rows.
                        row[f"{statistic}_{borough.lower()}"] = value
                row["brooklyn_gap_pct"] = _relative_gap(
                    summaries["Brooklyn"]["median_price_usd"], summaries["Manhattan"]["median_price_usd"])
                row["brooklyn_gap_excluding_largest_hosts_pct"] = _relative_gap(
                    summaries["Brooklyn"]["median_excluding_largest_host_usd"],
                    summaries["Manhattan"]["median_excluding_largest_host_usd"])
                row["brooklyn_gap_equal_host_medians_pct"] = _relative_gap(
                    summaries["Brooklyn"]["median_of_host_medians_usd"],
                    summaries["Manhattan"]["median_of_host_medians_usd"])
                row.update(_period_gates(row))
                if period == CONTEXT_PERIOD:
                    row["period_status"] = "historical_context"
                elif not principal:
                    row["period_status"] = "sensitivity_only"
                elif not row["sample_gate_met"]:
                    row["period_status"] = "insufficient_sample"
                elif row["period_criteria_met"]:
                    row["period_status"] = "meets_period_criteria"
                else:
                    row["period_status"] = "does_not_meet_period_criteria"
                rows.append(row)

    decision = _decide_main_replication(rows)
    table = pd.DataFrame(rows)
    output = root / "results"
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    table_path = tables / "temporal_market_replication.csv"
    table.to_csv(table_path, index=False, float_format="%.8f")
    protocol_path = root / "docs" / "TEMPORAL_EVALUATION_PROTOCOL.md"
    summary = {
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_document": "docs/TEMPORAL_EVALUATION_PROTOCOL.md",
        "protocol_document_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()
                                    if protocol_path.exists() else None,
        "principal_comparison": {
            "boroughs": ["Manhattan", "Brooklyn"],
            "room_type": "Entire home/apt", "bedrooms": 1, "declared_capacity": 2,
            "minimum_nights": 30, "maximum_station_distance_m": 500,
            "min_listings_per_borough": MIN_LISTINGS_PER_BOROUGH,
            "min_hosts_per_borough": MIN_HOSTS_PER_BOROUGH,
            "listing_median_brooklyn_at_least_pct_lower": MIN_LOWER_PRICE_PCT,
            "largest_host_exclusion_requires_same_lower_direction": True,
        },
        "decision": decision,
        "table": str(table_path.relative_to(root)),
        "periods_evaluated": [period for period in (CONTEXT_PERIOD, *FUTURE_PERIODS) if period in period_frames],
        "rows": table.replace({np.nan: None}).to_dict(orient="records"),
        "interpretation": [
            "Both fixed future periods are required; October context never counts toward replication.",
            "The market benchmark uses all eligible listings in its fixed product category, not the separate novel-host prediction cohort.",
            "Negative gaps mean Brooklyn's listing-weighted median is lower than Manhattan's; host sensitivities are separate estimands.",
            "Capacity and 800 m comparisons are sensitivities and cannot rescue a failed or undersupported principal comparison.",
            "Anonymized aggregate account concentration is reported; host identifiers are never exported.",
            "Common category filters do not match amenities, apartment area, listing quality or neighborhood composition.",
            "Temporal point replication does not establish a causal borough effect, achieved booking prices, occupancy or profit.",
        ],
    }
    (output / "temporal_market_replication.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary
