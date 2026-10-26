"""Frozen novelty and host-uniform selection rules for temporal evaluation.

Identifiers remain in memory. This module does not read price columns, export
identities, or use any outcome when deciding membership or selecting listings.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

HOST_SELECTION_PREFIX = "temporal-v1|"


def _identity_values(values: pd.Series) -> pd.Series:
    """Keep exact textual IDs, retaining missing/blank identities as unknown."""
    result = values.astype("string")
    return result.mask(result.str.strip().eq(""))


def read_raw_identity_sets(path: Path) -> tuple[set[str], set[str]]:
    """Read listing and host IDs from *all* raw rows, regardless of eligibility.

    Only these two columns are read; missing prices, zero/negative prices,
    hospitality status and minimum-stay requirements cannot narrow the prior
    identity sets. Missing IDs are omitted, without inventing substitute IDs.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw identity source does not exist: {path}")
    listing_ids: set[str] = set()
    host_ids: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["id", "host_id"], dtype=str,
                             chunksize=50_000, low_memory=False):
        listing_ids.update(_identity_values(chunk["id"]).dropna().tolist())
        host_ids.update(_identity_values(chunk["host_id"]).dropna().tolist())
    return listing_ids, host_ids


def _aligned_identities(frame: pd.DataFrame, hosts: pd.Series) -> tuple[pd.Series, pd.Series]:
    if "id" not in frame:
        raise ValueError("Temporal cohort features require a listing id column")
    if not isinstance(hosts, pd.Series):
        raise TypeError("Host IDs must be a Series aligned to the feature-frame index")
    if len(frame) != len(hosts) or not frame.index.equals(hosts.index):
        raise ValueError("Host IDs and feature rows must have identical order and index")
    return _identity_values(frame["id"]), _identity_values(hosts)


def strict_novel_mask(frame: pd.DataFrame, hosts: pd.Series,
                      prior_listing_ids: set[str], prior_host_ids: set[str]) -> np.ndarray:
    """Return a positional mask requiring both listing and host to be unseen.

    Unknown identities cannot establish novelty and therefore receive False.
    Prior sets must come from all raw historical rows, not priced/cohort rows.
    """
    listing_ids, host_ids = _aligned_identities(frame, hosts)
    novel = (listing_ids.notna() & host_ids.notna()
             & ~listing_ids.isin(prior_listing_ids) & ~host_ids.isin(prior_host_ids))
    return novel.to_numpy(dtype=bool, na_value=False)


def select_host_uniform(frame: pd.DataFrame, hosts: pd.Series) -> np.ndarray:
    """Select one row per host by the protocol's smallest listing-ID hash.

    The hash is SHA256(UTF8('temporal-v1|' + listing ID)). A lexical listing-ID
    tie-break makes even a hash collision deterministic. Returned integer
    positions are sorted so ``frame.iloc[positions]`` preserves input order.
    Prices and other features never enter selection. Call after novelty
    filtering; missing identities or duplicate listing IDs raise explicitly.
    """
    listing_ids, host_ids = _aligned_identities(frame, hosts)
    if listing_ids.isna().any() or host_ids.isna().any():
        raise ValueError("Host-uniform selection requires known listing and host identities")
    if not listing_ids.is_unique:
        raise ValueError("Host-uniform selection requires unique listing IDs")
    chosen: dict[str, tuple[bytes, str, int]] = {}
    for position, (listing_id, host_id) in enumerate(zip(listing_ids, host_ids)):
        digest = hashlib.sha256((HOST_SELECTION_PREFIX + listing_id).encode("utf-8")).digest()
        candidate = (digest, listing_id, position)
        existing = chosen.get(host_id)
        if existing is None or candidate[:2] < existing[:2]:
            chosen[host_id] = candidate
    return np.asarray(sorted(value[2] for value in chosen.values()), dtype=np.int64)
