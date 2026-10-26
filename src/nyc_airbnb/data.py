"""Source provenance, data validation and NYC geospatial feature preparation.

The Airbnb archive is pinned to the original snapshot, and the small original
station extract is bundled. Sources are hashed so a different snapshot cannot
silently replace the data used by the published results.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from scipy.spatial import cKDTree
from urllib3.util.retry import Retry

METRIC_CRS = "EPSG:32618"  # UTM 18N, metres; appropriate for New York City.
NYC_COUNTIES = {
    "005": "Bronx", "047": "Brooklyn", "061": "Manhattan",
    "081": "Queens", "085": "Staten Island",
}
SOURCE_SPECS = {
    "airbnb": {
        "filename": "listings.csv.gz",
        "url": "https://data.insideairbnb.com/united-states/ny/new-york-city/2025-10-01/data/listings.csv.gz",
        "period": "2025-10-01 Inside Airbnb archive; last_scraped dates also reported",
        "provenance": "User-supplied original Inside Airbnb detailed NYC extract",
        "download": True,
        "sha256": "a50efc7b11ef0a0786e38787ae2d4f47af1009943a1049503b676cc4b770fcbd",
    },
    "subway": {
        "filename": "subway.csv",
        "url": "https://data.ny.gov/resource/39hk-dx4f.csv?$limit=5000",
        "period": "Original user-supplied station extract; extract date unknown",
        "provenance": "User-supplied original MTA Subway Stations CSV",
        "download": False,
        "reference": "data/reference/subway.csv",
        "sha256": "4f10d7730b4dfe1737cb54426adccb8396346eceec2be9a22a58c9e0ccfdbc57",
    },
    "income": {
        "filename": "acs_income_2023.dat",
        "url": "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b19013.dat",
        "period": "2019-2023 ACS 5-year estimates; tract median household income",
        "provenance": "Census Bureau official table-based summary file B19013",
        "download": True,
        "sha256": "c9007017b7bb29ba758c77612ec9a4a0b897a4ed931d527754629866f580a110",
    },
    "population": {
        "filename": "acs_population_2023.dat",
        "url": "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b01003.dat",
        "period": "2019-2023 ACS 5-year estimates; tract population",
        "provenance": "Census Bureau official table-based summary file B01003",
        "download": True,
        "sha256": "24ae3f523b4c54332ce0a71ba534569685a8a729056f975915d860d1eb943565",
    },
    "tracts": {
        "filename": "tl_2023_36_tract.zip",
        "url": "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_36_tract.zip",
        "period": "2023 TIGER/Line census tract boundaries, New York State",
        "provenance": "Census Bureau official TIGER/Line shapefile",
        "download": True,
        "sha256": "010dfd8893c8a2a6482c7a0420b0113a8c0a0029137d0f904635014f39ffb918",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str | None = None) -> None:
    """Download atomically, with bounded connection/read timeouts and retries."""
    retry = Retry(total=3, backoff_factor=0.8,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    partial = destination.with_name(destination.name + ".part")
    try:
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=retry))
            with session.get(url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with partial.open("wb") as output:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            output.write(block)
        if partial.stat().st_size == 0:
            raise ValueError(f"Downloaded an empty source: {destination.name}")
        with partial.open("rb") as stream:
            prefix = stream.read(200).lstrip().lower()
        if prefix.startswith((b"<!doctype html", b"<html")):
            raise ValueError(f"Source returned HTML instead of data: {destination.name}")
        if expected_sha256 and sha256_file(partial) != expected_sha256:
            raise ValueError(f"Downloaded source does not match the pinned SHA256: {destination.name}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def fetch_sources(root: Path, offline: bool = False) -> tuple[dict[str, Path], dict]:
    """Resolve exact source files and record SHA256 hashes in data/raw/manifest.json.

    Existing hashes are checked on every run. Changing a cached file without
    also explicitly replacing its provenance record raises an error.
    """
    raw = Path(root) / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    records = previous.get("sources", {})
    paths: dict[str, Path] = {}
    manifest = {"schema_version": 1, "sources": {}}
    for key, spec in SOURCE_SPECS.items():
        path = raw / spec["filename"]
        downloaded = False
        if not path.exists() and spec.get("reference"):
            reference = Path(root) / spec["reference"]
            if reference.exists():
                shutil.copyfile(reference, path)
        if not path.exists():
            if offline:
                raise FileNotFoundError(f"Offline source missing: {path}")
            if not spec["download"]:
                raise FileNotFoundError(
                    f"Original {key} extract missing: {path}. Copy the original "
                    "file here; this workflow does not silently change snapshots."
                )
            _download(spec["url"], path, spec.get("sha256"))
            downloaded = True
        digest = sha256_file(path)
        if spec.get("sha256") and spec["sha256"] != digest:
            raise ValueError(f"Cached source differs from pinned SHA256: {path.name}")
        old = records.get(key, {})
        if old.get("sha256") and old["sha256"] != digest:
            raise ValueError(f"Cached source hash mismatch: {path.name}")
        record = {
            "filename": path.name, "url": spec["url"], "period": spec["period"],
            "provenance": spec["provenance"], "sha256": digest,
            "bytes": path.stat().st_size,
            "registered_at_utc": old.get("registered_at_utc", datetime.now(timezone.utc).isoformat()),
            "downloaded_by_pipeline": old.get("downloaded_by_pipeline", downloaded),
        }
        manifest["sources"][key] = record
        paths[key] = path
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return paths, manifest


def parse_price(values: pd.Series) -> pd.Series:
    """Parse numeric or dollar/comma price values; invalid values remain missing."""
    cleaned = values.astype("string").str.replace(r"[$,]", "", regex=True).str.strip()
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def clean_acs_values(values: pd.Series, *, positive_only: bool = False) -> pd.Series:
    """Exclude Census negative sentinel codes without treating missing as zero."""
    result = pd.to_numeric(values, errors="coerce").astype(float)
    return result.where(result > 0 if positive_only else result >= 0)


def read_acs_table(path: Path, table: str) -> pd.DataFrame:
    """Read only NYC census tracts from a national pipe-delimited ACS table."""
    selected = []
    for chunk in pd.read_csv(path, sep="|", dtype=str, chunksize=50_000,
                             encoding="utf-8", low_memory=False):
        columns = {column.upper(): column for column in chunk.columns}
        geo_col = columns.get("GEO_ID")
        estimate = columns.get("E001", columns.get(f"{table.upper()}_E001"))
        margin = columns.get("M001", columns.get(f"{table.upper()}_M001"))
        if not geo_col or not estimate:
            raise ValueError(f"Unexpected ACS schema for {path.name}: {list(chunk.columns)}")
        rows = chunk[chunk[geo_col].str.startswith("1400000US36", na=False)].copy()
        rows["GEOID"] = rows[geo_col].str[-11:]
        rows = rows[rows["GEOID"].str[2:5].isin(NYC_COUNTIES)]
        result = pd.DataFrame({"GEOID": rows["GEOID"], "estimate": rows[estimate]})
        result["moe"] = rows[margin] if margin else np.nan
        selected.append(result)
    result = pd.concat(selected, ignore_index=True)
    if result["GEOID"].duplicated().any():
        raise ValueError(f"Duplicate NYC GEOIDs in ACS table {table}")
    result["estimate"] = clean_acs_values(result["estimate"], positive_only=table.upper() == "B19013")
    result["moe"] = clean_acs_values(result["moe"])
    return result


def read_stations(path: Path) -> tuple[pd.DataFrame, dict]:
    """Exclude Staten Island Railway, then deduplicate IDs and coordinates."""
    raw = pd.read_csv(path, dtype=str)
    normalized = {column.lower().replace("_", " ").strip(): column for column in raw.columns}

    def column(*variants: str) -> str | None:
        return next((normalized[v] for v in variants if v in normalized), None)

    lat = column("gtfs latitude", "gtfs latit", "latitude", "lat")
    lon = column("gtfs longitude", "gtfs long", "longitude", "lon", "lng")
    name = column("stop name", "station name", "name")
    identity = column("station id")
    division = column("division")
    if not lat or not lon or not name:
        raise ValueError("Station file requires coordinates and station names")
    stations = raw.copy()
    sir = stations[division].str.upper().eq("SIR") if division else pd.Series(False, index=stations.index)
    stations = stations[~sir].copy()
    stats = {"station_rows_raw": len(raw), "sir_rows_excluded": int(sir.sum())}
    if identity:
        stations = stations.drop_duplicates(identity)
    stats["station_ids_unique_after_sir_exclusion"] = len(stations)
    result = pd.DataFrame({
        "station_name": stations[name].fillna("Unnamed station"),
        "latitude": pd.to_numeric(stations[lat], errors="coerce"),
        "longitude": pd.to_numeric(stations[lon], errors="coerce"),
    }).dropna(subset=["latitude", "longitude"])
    result = result[result["latitude"].between(40, 42) & result["longitude"].between(-75, -72)]
    result = result.sort_values(["station_name", "latitude", "longitude"], kind="stable")
    result = result.drop_duplicates(["latitude", "longitude"]).reset_index(drop=True)
    if result.empty:
        raise ValueError("No valid NYC subway locations remained")
    stats["unique_station_locations"] = len(result)
    return result, stats


def attach_tracts(points: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    """Join tract attributes without expanding listings; accept unique boundary matches."""
    if tracts["GEOID"].duplicated().any():
        raise ValueError("Tract GEOIDs must be unique")
    if not points.index.is_unique:
        raise ValueError("Listing index must be unique before joining tracts")
    joined = gpd.sjoin(points, tracts, how="left", predicate="within")
    if not joined.index.is_unique:
        raise ValueError("Overlapping tract interiors expanded listing rows")
    unmatched = joined["GEOID"].isna()
    boundary_count = ambiguous_count = 0
    if unmatched.any():
        candidates = gpd.sjoin(points.loc[unmatched], tracts, how="inner", predicate="intersects")
        if not candidates.empty:
            counts = candidates.groupby(level=0).size()
            unique = candidates.loc[candidates.index.isin(counts[counts == 1].index)]
            ambiguous_count = int((counts > 1).sum())
            boundary_count = len(unique)
            for attribute in tracts.columns.difference(["geometry"]):
                joined.loc[unique.index, attribute] = unique[attribute]
    joined = joined.drop(columns="index_right", errors="ignore")
    if len(joined) != len(points):
        raise AssertionError("Spatial join changed the listing count")
    return joined, {"unique_boundary_matches": boundary_count, "ambiguous_boundary_listings": ambiguous_count}


def accessibility_features(points: gpd.GeoDataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Nearest station and count of unique station locations within 800 metres."""
    projected = points.to_crs(METRIC_CRS)
    station_geo = gpd.GeoDataFrame(stations.copy(), geometry=gpd.points_from_xy(
        stations["longitude"], stations["latitude"]), crs="EPSG:4326").to_crs(METRIC_CRS)
    station_xy = np.column_stack([station_geo.geometry.x, station_geo.geometry.y])
    point_xy = np.column_stack([projected.geometry.x, projected.geometry.y])
    tree = cKDTree(station_xy)
    distances, indices = tree.query(point_xy, k=1)
    result = pd.DataFrame(index=points.index)
    result["dist_km"] = distances / 1000
    result["nearest_station"] = stations.iloc[indices]["station_name"].to_numpy()
    result["stations_within_800m"] = tree.query_ball_point(point_xy, r=800, return_length=True)
    result["x_m"] = point_xy[:, 0]
    result["y_m"] = point_xy[:, 1]
    # Geographic groups for evaluation, not predictive features.
    result["spatial_block"] = (np.floor(point_xy[:, 0] / 2000).astype(int).astype(str)
                               + "_" + np.floor(point_xy[:, 1] / 2000).astype(int).astype(str))
    return result


def prepare_data(root: Path, offline: bool = False) -> tuple[pd.DataFrame, gpd.GeoDataFrame, pd.DataFrame, dict]:
    """Prepare positive-priced listings in NYC tracts, without price-tail trimming."""
    paths, manifest = fetch_sources(Path(root), offline=offline)
    raw = pd.read_csv(paths["airbnb"], compression="gzip", low_memory=False, dtype={"id": str})
    required = {"id", "price", "latitude", "longitude", "room_type"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Airbnb input missing required fields: {sorted(required - set(raw.columns))}")
    quality: dict = {
        "source_manifest": manifest, "raw_listings": len(raw),
        "price_definition": "Advertised nightly listing price, USD; not booked revenue or property sale value",
        "distance_definition": "Straight-line metres in EPSG:32618 to unique MTA subway station locations; SIR excluded",
        "price_tail_filter": "None: retain every finite positive price",
    }
    if "last_scraped" in raw:
        dates = pd.to_datetime(raw["last_scraped"], errors="coerce")
        quality["listing_scrape_date_min"] = str(dates.min().date())
        quality["listing_scrape_date_max"] = str(dates.max().date())
    raw["price"] = parse_price(raw["price"])
    quality["missing_or_invalid_price"] = int(raw["price"].isna().sum())
    quality["nonpositive_price"] = int((raw["price"] <= 0).sum())
    quality["duplicate_listing_ids_removed"] = int(raw["id"].duplicated().sum())
    rows = raw.drop_duplicates("id", keep="first").copy()
    rows = rows[rows["id"].notna() & np.isfinite(rows["price"]) & (rows["price"] > 0)].copy()
    quality["positive_price_unique_listings"] = len(rows)
    for name in ["latitude", "longitude"]:
        rows[name] = pd.to_numeric(rows[name], errors="coerce")
    rows = rows[rows["latitude"].between(-90, 90) & rows["longitude"].between(-180, 180)].copy()
    quality["valid_coordinate_listings"] = len(rows)
    safe_columns = ["id", "price", "latitude", "longitude", "room_type", "property_type",
                    "neighbourhood_cleansed", "accommodates", "bedrooms", "beds", "bathrooms",
                    "minimum_nights", "number_of_reviews", "reviews_per_month", "review_scores_rating",
                    "calculated_host_listings_count", "availability_365", "instant_bookable"]
    features = rows[[column for column in safe_columns if column in rows]].copy()
    numeric = ["accommodates", "bedrooms", "beds", "bathrooms", "minimum_nights", "number_of_reviews",
               "reviews_per_month", "review_scores_rating", "calculated_host_listings_count", "availability_365"]
    for name in numeric:
        features[name] = pd.to_numeric(rows[name], errors="coerce") if name in rows else np.nan
        features[name] = features[name].where(features[name] >= 0)
    if "bathrooms_text" in rows:
        fallback = pd.to_numeric(rows["bathrooms_text"].astype("string").str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
        features["bathrooms"] = features["bathrooms"].fillna(fallback)
    if "instant_bookable" in features:
        features["instant_bookable"] = features["instant_bookable"].map({"t": 1, "f": 0, True: 1, False: 0})
    for name in ["room_type", "property_type", "neighbourhood_cleansed"]:
        if name not in features:
            features[name] = "Unknown"
        features[name] = features[name].fillna("Unknown").astype(str)

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
    quality["nyc_tracts"] = len(tracts)
    quality["tracts_missing_income"] = int(tracts["median_household_income"].isna().sum())
    points = gpd.GeoDataFrame(features, geometry=gpd.points_from_xy(
        features["longitude"], features["latitude"]), crs="EPSG:4326")
    joined, join_quality = attach_tracts(points, tracts.drop(columns="COUNTYFP"))
    quality.update(join_quality)
    quality["unmatched_nyc_tract_listings"] = int(joined["GEOID"].isna().sum())
    joined = joined[joined["GEOID"].notna()].copy().reset_index(drop=True)
    if joined.empty:
        raise ValueError("No positive-priced listings matched NYC census tracts")
    stations, station_quality = read_stations(paths["subway"])
    quality.update(station_quality)
    accessibility = accessibility_features(joined, stations)
    for name in accessibility:
        joined[name] = accessibility[name]
    # Predictive metrics and inverse transforms use log1p throughout. OLS
    # inference constructs its own natural-log target for percent associations.
    joined["log_price"] = np.log1p(joined["price"])
    joined["log_minimum_nights"] = np.log1p(joined["minimum_nights"])
    joined["log_review_count"] = np.log1p(joined["number_of_reviews"])
    joined["log_host_listings_count"] = np.log1p(joined["calculated_host_listings_count"])
    result = pd.DataFrame(joined.drop(columns="geometry")).sort_values("id", kind="stable").reset_index(drop=True)
    quality["analysis_listings"] = len(result)
    quality["listings_missing_income"] = int(result["median_household_income"].isna().sum())
    quality["missing_income_pct"] = float(result["median_household_income"].isna().mean() * 100)
    quality["missing_pct"] = {name: float(result[name].isna().mean() * 100) for name in numeric + ["median_household_income"]}
    quality["notes"] = [
        "Income/population describe census tracts and do not measure individual hosts or guests.",
        "ACS 2019-2023 estimates precede the listing snapshot; income margins of error are retained.",
        "Station extract date is unknown. Straight-line access is not walking distance or travel time.",
        "Staten Island listings can be far from the subway because SIR stations are excluded.",
        "Listings without prices are excluded; observed prices may not represent all NYC listings.",
        "Coordinates are approximate, so tract membership and station distance have measurement error.",
    ]
    return result, tracts, stations, quality
