"""Frozen training-only comparable medians for temporal protocol version 1.

Both outputs share the first supported cell in the protocol's fixed hierarchy.
The listing output weights listings equally. The host output takes a raw-dollar
median within each host/cell, then a median across those host medians.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


UNKNOWN = "Unknown"
MIN_LISTINGS = 20
MIN_HOSTS = 5
KEY_LEVELS = (
    ("borough", "neighborhood", "room_type", "capacity", "bedrooms", "minimum_stay"),
    ("borough", "neighborhood", "room_type", "capacity", "bedrooms"),
    ("borough", "neighborhood", "room_type", "capacity"),
    ("borough", "room_type", "capacity", "bedrooms", "minimum_stay"),
    ("borough", "room_type", "capacity", "bedrooms"),
    ("borough", "room_type", "capacity"),
    ("borough", "room_type"),
    ("room_type",),
)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return frame[name] if name in frame else pd.Series(np.nan, index=frame.index)


def _string_key(values: pd.Series) -> pd.Series:
    # Preserve recorded case and spelling; remove only surrounding whitespace.
    result = values.astype("string").str.strip()
    return result.fillna(UNKNOWN).replace("", UNKNOWN).astype(str)


def _bounded_count_key(values: pd.Series, exact: tuple[int, ...], upper: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(UNKNOWN, index=values.index, dtype=object)
    finite = pd.Series(np.isfinite(numeric), index=values.index)
    for number in exact:
        result.loc[finite & numeric.eq(number)] = str(number)
    result.loc[finite & numeric.ge(upper)] = f"{upper}+"
    return result


def normalize_comparable_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the protocol's fixed, price-independent key normalization."""
    keys = pd.DataFrame(index=frame.index)
    keys["borough"] = _string_key(_column(frame, "borough"))
    keys["neighborhood"] = _string_key(_column(frame, "neighbourhood_cleansed"))
    keys["room_type"] = _string_key(_column(frame, "room_type"))
    keys["capacity"] = _bounded_count_key(_column(frame, "accommodates"), tuple(range(1, 7)), 7)
    keys["bedrooms"] = _bounded_count_key(_column(frame, "bedrooms"), (0, 1, 2), 3)
    nights = pd.to_numeric(_column(frame, "minimum_nights"), errors="coerce")
    finite = pd.Series(np.isfinite(nights), index=frame.index)
    keys["minimum_stay"] = UNKNOWN
    for mask, label in [
        (nights.eq(30), "30"), (nights.eq(31), "31"),
        (nights.between(32, 59), "32-59"), (nights.between(60, 89), "60-89"),
        (nights.ge(90), "90+"),
    ]:
        keys.loc[finite & mask, "minimum_stay"] = label
    return keys


class ComparableBaselines:
    """Two frozen comparable-price estimates fitted exclusively on training rows.

    ``host_ids`` is positional and must match the supplied training row order.
    Global fallback always exists and does not require cell-support thresholds.
    No host identifiers or future prices enter prediction.
    """

    def fit(self, frame: pd.DataFrame, host_ids) -> "ComparableBaselines":
        if len(frame) == 0:
            raise ValueError("Comparable baselines require nonempty training data")
        if "price" not in frame:
            raise ValueError("Training prices are required")
        price = pd.to_numeric(frame["price"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(price).all() or not (price > 0).all():
            raise ValueError("Training prices must be finite and positive")
        if "id" in frame and not frame["id"].is_unique:
            raise ValueError("Training listing IDs must be unique")
        hosts = np.asarray(host_ids)
        if len(hosts) != len(frame):
            raise ValueError("Host IDs must match training row count and order")
        hosts = pd.Series(hosts, dtype="string")
        if hosts.isna().any() or hosts.str.strip().eq("").any():
            raise ValueError("Every training listing requires a host ID")

        training = normalize_comparable_keys(frame).reset_index(drop=True)
        training["_price"] = price
        training["_host"] = hosts.astype(str).to_numpy()
        self.global_listing_median_ = float(np.median(price))
        self.global_host_median_ = float(training.groupby("_host")["_price"].median().median())
        self.training_n_ = len(training)
        self.training_hosts_ = int(training["_host"].nunique())
        self.cells_: list[dict[tuple, tuple[float, float]]] = []
        self.support_: list[pd.DataFrame] = []
        for level in KEY_LEVELS:
            columns = list(level)
            grouped = training.groupby(columns, sort=True, observed=True)
            support = grouped.agg(
                n_listings=("_price", "size"), n_hosts=("_host", "nunique"),
                listing_median=("_price", "median"),
            )
            per_host = training.groupby(columns + ["_host"], sort=True, observed=True)["_price"].median().reset_index()
            support["host_median"] = per_host.groupby(columns, sort=True, observed=True)["_price"].median()
            support = support[(support["n_listings"] >= MIN_LISTINGS) & (support["n_hosts"] >= MIN_HOSTS)]
            self.support_.append(support.reset_index())
            mapping = {}
            for row in support.reset_index().itertuples(index=False, name=None):
                key = tuple(row[:len(columns)])
                # Remaining fields are listing count, host count, and the two medians.
                mapping[key] = (float(row[len(columns) + 2]), float(row[len(columns) + 3]))
            self.cells_.append(mapping)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "cells_"):
            raise RuntimeError("Fit comparable baselines before prediction")
        keys = normalize_comparable_keys(frame)
        listing = np.full(len(frame), self.global_listing_median_, dtype=float)
        host = np.full(len(frame), self.global_host_median_, dtype=float)
        levels = np.full(len(frame), 9, dtype=int)
        unresolved = np.ones(len(frame), dtype=bool)
        for level_number, (columns, mapping) in enumerate(zip(KEY_LEVELS, self.cells_), start=1):
            if not unresolved.any():
                break
            positions = np.flatnonzero(unresolved)
            for position, key in zip(positions, keys.iloc[positions][list(columns)].itertuples(index=False, name=None)):
                match = mapping.get(tuple(key))
                if match is not None:
                    listing[position], host[position] = match
                    levels[position] = level_number
                    unresolved[position] = False
        return pd.DataFrame({
            "comparable_listing_median": listing,
            "comparable_host_median": host,
            "comparable_level": levels,
        }, index=frame.index)
