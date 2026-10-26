"""Portable figures and a geographic map built only from reviewed run outputs."""

from __future__ import annotations

from html import escape
from pathlib import Path
import re
import textwrap

import folium
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, NullFormatter, PercentFormatter
import numpy as np
import pandas as pd
from branca.colormap import LinearColormap

METRIC_CRS = "EPSG:32618"
BLUE = "#23657d"
ORANGE = "#c87736"
INK = "#26333b"
GRID = "#e3e8eb"
SEED = 42
FIGURE_FILES = {
    "validation": "model_validation.png",
    "association": "subway_association.png",
    "segments": "distance_by_room_type.png",
    "geography": "listing_geography.png",
}


def _label(value: object) -> str:
    return str(value).replace("_", " ").strip().capitalize()


def _table(root: Path, name: str, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(root / "results" / "tables" / name)
    missing = required.difference(frame.columns)
    if missing or frame.empty:
        raise ValueError(f"Invalid presentation input {name}: missing {sorted(missing)} or empty")
    return frame


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.labelcolor": INK, "axes.edgecolor": "#98a6ae",
        "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.axisbelow": True,
    })


def _save(fig: plt.Figure, path: Path, note: str) -> None:
    fig.text(0.02, 0.015, note, ha="left", va="bottom", fontsize=9, color="#52626c")
    fig.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def _validation_figure(root: Path, path: Path) -> None:
    data = _table(root, "model_metrics.csv", {"validation", "model", "n", "mae_usd"})
    if data.duplicated(["validation", "model"]).any():
        raise ValueError("Model metrics require one row per validation/model")
    if not np.isfinite(data["mae_usd"]).all() or (data["mae_usd"] < 0).any():
        raise ValueError("MAE must be finite and nonnegative")
    models = list(dict.fromkeys(data["model"]))
    validations = list(dict.fromkeys(data["validation"]))
    values = data.pivot(index="model", columns="validation", values="mae_usd").reindex(models)
    height = max(5.4, len(models) * 0.8 + 2)
    fig, ax = plt.subplots(figsize=(11, height))
    fig.subplots_adjust(left=0.28, right=0.95, top=0.78, bottom=0.18)
    positions = np.arange(len(models))
    bar_height = min(0.32, 0.75 / len(validations))
    colors = [BLUE, ORANGE, "#7c8060", "#95657e"]
    for index, validation in enumerate(validations):
        offset = (index - (len(validations) - 1) / 2) * bar_height
        y = positions + offset
        series = values[validation]
        bars = ax.barh(y, series, height=bar_height * 0.92,
                       label=_label(validation), color=colors[index % len(colors)])
        for bar, value in zip(bars, series):
            if pd.notna(value):
                ax.text(value + values.max().max() * 0.012,
                        bar.get_y() + bar.get_height() / 2,
                        f"${value:,.2f}", va="center", fontsize=10)
    ax.set_yticks(positions, [textwrap.fill(_label(model), 26) for model in models])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values.max().max() * 1.22, 1))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.set_xlabel("Mean absolute error per listing (USD); lower is better")
    ax.grid(axis="x", color=GRID)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False,
              ncol=min(3, len(validations)), borderaxespad=0)
    fig.suptitle("Price prediction under random and spatial validation", x=0.02,
                 y=0.96, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.02, 0.885, "Out-of-fold dollar errors; October 2025 primary sample excluding hospitality listings.",
             ha="left", fontsize=11)
    n_low, n_high = int(data["n"].min()), int(data["n"].max())
    n_note = f"n={n_low:,}" if n_low == n_high else f"n={n_low:,}–{n_high:,}; see the source table for each model"
    _save(fig, path, f"Evaluation rows: {n_note}. Source: results/tables/model_metrics.csv.\n"
          "Spatial validation holds out geographic blocks; it does not test a future snapshot.")


def _association_figure(root: Path, path: Path) -> None:
    data = _table(root, "coefficient_estimates.csv", {
        "specification", "n", "tract_clusters", "percent_per_500m", "ci_lower_pct", "ci_upper_pct"})
    point = pd.to_numeric(data["percent_per_500m"])
    lower = pd.to_numeric(data["ci_lower_pct"])
    upper = pd.to_numeric(data["ci_upper_pct"])
    if not np.isfinite(np.column_stack([point, lower, upper])).all():
        raise ValueError("Coefficient points and confidence intervals must be finite")
    if ((lower > point) | (upper < point)).any():
        raise ValueError("Confidence interval does not contain its coefficient estimate")
    fig, ax = plt.subplots(figsize=(11.8, max(5.4, len(data) * 0.9 + 2.2)))
    fig.subplots_adjust(left=0.32, right=0.78, top=0.76, bottom=0.19)
    y = np.arange(len(data))
    ax.errorbar(point, y, xerr=np.vstack([point - lower, upper - point]),
                fmt="o", color=BLUE, ecolor=BLUE, capsize=4, markersize=7, linewidth=2)
    labels = [f"{textwrap.fill(_label(row.specification), 29)}\n"
              f"n={int(row.n):,}; tracts={int(row.tract_clusters):,}"
              for row in data.itertuples()]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#61717a", linewidth=1.1, linestyle="--")
    lo, hi = min(float(lower.min()), 0), max(float(upper.max()), 0)
    pad = max((hi - lo) * 0.12, 0.5)
    ax.set_xlim(lo - pad, hi + pad)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Estimated price difference per 500 m farther from a station")
    ax.grid(axis="x", color=GRID)
    for index, row in enumerate(data.itertuples()):
        ax.text(1.035, index,
                f"{row.percent_per_500m:+.1f}%\n[{row.ci_lower_pct:+.1f}%, {row.ci_upper_pct:+.1f}%]",
                transform=ax.get_yaxis_transform(), va="center", fontsize=10)
    fig.suptitle("Subway distance and advertised nightly price", x=0.02, y=0.96,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(0.02, 0.88, "Points transform ln(price) coefficients; bars show 95% confidence intervals.",
             ha="left", fontsize=11)
    _save(fig, path, "Confidence intervals are clustered by census tract. Each specification uses its reported sample.\n"
          "Observational associations, not causal transit effects. Source: results/tables/coefficient_estimates.csv.")


def _distance_order(value: object) -> tuple[float, str]:
    label = str(value)
    numbers = re.findall(r"\d+(?:\.\d+)?", label)
    if not numbers:
        return float("inf"), label
    number = float(numbers[0])
    first = re.search(r"\d+(?:\.\d+)?\s*(km|m)?", label.lower())
    units = re.findall(r"(?<![a-z])(km|m)\b", label.lower())
    unit = first.group(1) if first and first.group(1) else units[0] if units else "m"
    if unit == "km":
        number *= 1000
    return number, label


def _segments_figure(root: Path, path: Path) -> None:
    data = _table(root, "distance_segments.csv", {"room_type", "distance_band", "n", "median_price"})
    rooms = list(dict.fromkeys(data["room_type"]))
    bands = sorted(data["distance_band"].dropna().unique(), key=_distance_order)
    columns = min(2, len(rooms))
    rows = int(np.ceil(len(rooms) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 3.9 * rows + 1.9),
                             squeeze=False, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.14,
                        hspace=0.52, wspace=0.18)
    finite_prices = pd.to_numeric(data["median_price"], errors="coerce")
    maximum = max(float(finite_prices.max()) * 2.5, 1)
    minimum = max(float(finite_prices[finite_prices > 0].min()) / 1.7, 0.1)
    for ax, room in zip(axes.flat, rooms):
        subset = data[data["room_type"] == room].set_index("distance_band").reindex(bands)
        x = np.arange(len(bands))
        ax.scatter(x, subset["median_price"], color=BLUE, s=48, edgecolor="white", linewidth=0.6)
        for index, row in enumerate(subset.itertuples()):
            if pd.notna(row.median_price):
                ax.text(index, row.median_price * 1.22,
                        f"${row.median_price:,.0f}\nn={int(row.n):,}",
                        ha="center", va="bottom", fontsize=9)
        ax.set_title(str(room), loc="left", fontsize=12, fontweight="bold")
        ax.set_xticks(x, [textwrap.fill(str(band), 14) for band in bands], fontsize=9)
        ax.set_yscale("log")
        ax.set_ylim(minimum, maximum)
        ticks = [base * (10 ** power) for power in range(-1, 7) for base in (1, 2, 5)
                 if minimum <= base * (10 ** power) <= maximum]
        ax.set_yticks(ticks)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("Median nightly price (USD; log scale)")
        ax.set_xlim(-0.45, len(bands) - 0.55)
        ax.set_xlabel("Straight-line station distance")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
        ax.grid(axis="y", color=GRID)
    for index, ax in enumerate(list(axes.flat)[len(rooms):]):
        ax.set_axis_off()
        if index == 0:
            ax.text(0.02, 0.85, "Population and reading guide", transform=ax.transAxes,
                    fontsize=12, fontweight="bold")
            reading = (f"{int(data['n'].sum()):,} listings across {len(data):,} observed groups.\n"
                       f"{int((data['n'] < 30).sum()):,} groups have fewer than 30 listings.\n\n"
                       "All panels use the same logarithmic price scale.\n"
                       "Distances are straight-line measurements.\n"
                       "Medians do not adjust for amenities or neighborhood.\n"
                       "The primary sample does not trim price tails.")
            ax.text(0.02, 0.73, reading, transform=ax.transAxes, fontsize=11,
                    va="top", linespacing=1.7, color="#52626c")
    fig.suptitle("Advertised prices by subway access and room type", x=0.02, y=0.975,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(0.02, 0.91, "Unadjusted medians on a common logarithmic price scale; labels show group sizes.",
             ha="left", fontsize=11)
    _save(fig, path, "Primary sample excludes hotel/hostel/resort listings; group medians do not isolate a transit effect.\n"
          "October 2025 listing snapshot. Source: results/tables/distance_segments.csv.")


def _geography_figure(root: Path, path: Path) -> None:
    listings = pd.read_csv(root / "data" / "processed" / "listings.csv", dtype={"id": str, "GEOID": str})
    tracts = gpd.read_file(root / "data" / "processed" / "tracts.geojson").to_crs(METRIC_CRS)
    valid = listings[np.isfinite(listings["latitude"]) & np.isfinite(listings["longitude"])
                     & np.isfinite(listings["dist_km"])].copy()
    sample = valid.sample(min(8000, len(valid)), random_state=SEED)
    points = gpd.GeoDataFrame(sample, geometry=gpd.points_from_xy(
        sample["longitude"], sample["latitude"]), crs="EPSG:4326").to_crs(METRIC_CRS)
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(left=0.02, right=0.87, top=0.86, bottom=0.12)
    tracts.plot(ax=ax, facecolor="#f3f5f6", edgecolor="#d8dfe3", linewidth=0.25)
    scatter = ax.scatter(points.geometry.x, points.geometry.y,
                         c=points["dist_km"], cmap="cividis", norm=Normalize(0, 2),
                         s=6, linewidths=0, alpha=0.8, rasterized=True)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.04, pad=0.02, shrink=0.7, extend="max")
    colorbar.set_label("Straight-line distance to subway (km)\ncolors capped at 2 km")
    ax.set_aspect("equal")
    bounds = tracts.total_bounds
    scale_x = bounds[0] + (bounds[2] - bounds[0]) * 0.07
    scale_y = bounds[1] + (bounds[3] - bounds[1]) * 0.035
    ax.plot([scale_x, scale_x + 5000], [scale_y, scale_y], color=INK, linewidth=3)
    ax.text(scale_x + 2500, scale_y + 550, "5 km", ha="center", fontsize=10)
    ax.annotate("N", xy=(0.92, 0.91), xytext=(0.92, 0.84),
                xycoords="axes fraction", textcoords="axes fraction", ha="center",
                arrowprops={"arrowstyle": "-|>", "color": INK}, fontsize=11)
    ax.set_axis_off()
    fig.suptitle("Subway access across NYC's primary listing sample", x=0.02, y=0.96,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(0.02, 0.905, f"{len(points):,} listing points shown from {len(valid):,} valid locations; "
             "NYC census tract boundaries.", ha="left", fontsize=11)
    _save(fig, path, "October 2025 listings; coordinates may be displaced by about 150 m. UTM 18N projection.\n"
          "Subway excludes Staten Island Railway. Source: Inside Airbnb, MTA, Census TIGER/Line 2023.")


def create_figures(root: Path | str) -> dict[str, Path]:
    """Render four figures from saved tables and processed geographic inputs."""
    root = Path(root)
    listing_count = len(pd.read_csv(root / "data" / "processed" / "listings.csv", usecols=["id"]))
    model_rows = _table(root, "model_metrics.csv", {"n"})
    if not model_rows["n"].eq(listing_count).all():
        raise ValueError("Saved model tables and processed listings have different populations; rerun the pipeline before plotting")
    destination = root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    _style()
    paths = {key: destination / filename for key, filename in FIGURE_FILES.items()}
    _validation_figure(root, paths["validation"])
    _association_figure(root, paths["association"])
    _segments_figure(root, paths["segments"])
    _geography_figure(root, paths["geography"])
    return paths


def _safe(value: object) -> str:
    return escape("Unknown" if pd.isna(value) else str(value), quote=True)


def create_map(root: Path | str) -> Path:
    """Write a Folium map using WGS84 coordinates and a deterministic sample.

    The HTML embeds reviewed geometry and marker values. Base tiles and Folium's
    browser libraries require internet access when viewing the interactive map.
    """
    root = Path(root)
    listings = pd.read_csv(root / "data" / "processed" / "listings.csv", dtype={"id": str, "GEOID": str})
    tracts = gpd.read_file(root / "data" / "processed" / "tracts.geojson").to_crs("EPSG:4326")
    locations = listings[np.isfinite(listings["latitude"]) & np.isfinite(listings["longitude"])]
    locations = locations[locations["latitude"].between(-90, 90)
                          & locations["longitude"].between(-180, 180)]
    if locations.empty:
        raise ValueError("No valid latitude/longitude points for the map")
    sample = locations.sample(min(2000, len(locations)), random_state=SEED)
    map_view = folium.Map(location=[40.72, -73.94], zoom_start=11, tiles="OpenStreetMap",
                          control_scale=True, prefer_canvas=True)
    income = pd.to_numeric(tracts["median_household_income"], errors="coerce")
    valid_income = tracts[income.notna() & (income > 0)].copy()
    if not valid_income.empty:
        low = float(valid_income["median_household_income"].min())
        high = float(valid_income["median_household_income"].max())
        color = LinearColormap(["#edf4f4", "#9fc5ce", "#477f94", "#164b64"],
                               vmin=low, vmax=max(high, low + 1),
                               caption="Tract median household income, 2023 USD (ACS 2019–2023)")
        fields = ["GEOID", "median_household_income", "geometry"]
        if "borough" in valid_income:
            fields.insert(1, "borough")
        # Popup/tooltip text contains only controlled numeric fields and tract IDs.
        folium.GeoJson(
            valid_income[fields].to_json(), name="ACS tract income", show=False,
            style_function=lambda feature: {
                "fillColor": color(feature["properties"]["median_household_income"]),
                "color": "#64747d", "weight": 0.35, "fillOpacity": 0.6},
            highlight_function=lambda _: {"weight": 1.2, "fillOpacity": 0.8},
            tooltip=folium.GeoJsonTooltip(
                fields=[field for field in fields if field != "geometry"],
                aliases=["Census tract" if field == "GEOID" else
                         "Median household income (2023 USD)" if field == "median_household_income"
                         else "Borough" for field in fields if field != "geometry"],
                localize=True, sticky=False),
        ).add_to(map_view)
        color.add_to(map_view)
    listing_layer = folium.FeatureGroup(name=f"Listings: seeded sample of {len(sample):,}", show=True)
    for row in sample.itertuples():
        price = float(row.price)
        distance = float(row.dist_km) * 1000
        popup = (f"<b>Listing {_safe(row.id)}</b><br>"
                 f"Room type: {_safe(row.room_type)}<br>"
                 f"Borough: {_safe(row.borough)}<br>"
                 f"Advertised nightly price: ${price:,.0f}<br>"
                 f"Nearest subway: {distance:,.0f} m (straight line)<br>"
                 f"Snapshot: October 2025")
        tooltip = (f"{_safe(row.room_type)} · ${price:,.0f}/night · "
                   f"{distance:,.0f} m to subway · Oct 2025")
        folium.CircleMarker(location=[float(row.latitude), float(row.longitude)],
                            radius=3, color=ORANGE, weight=0.5, fill=True,
                            fill_color=ORANGE, fill_opacity=0.65,
                            popup=folium.Popup(popup, max_width=330),
                            tooltip=tooltip).add_to(listing_layer)
    listing_layer.add_to(map_view)
    station_path = root / "data" / "processed" / "stations.csv"
    if station_path.exists():
        stations = pd.read_csv(station_path)
        station_layer = folium.FeatureGroup(name="Subway stations (SIR excluded)", show=False)
        for row in stations.itertuples():
            if not np.isfinite(row.latitude) or not np.isfinite(row.longitude):
                continue
            station_name = _safe(getattr(row, "station_name", "Subway station"))
            folium.CircleMarker([float(row.latitude), float(row.longitude)], radius=3,
                                color=INK, fill=True, fill_opacity=0.9,
                                tooltip=station_name, popup=station_name).add_to(station_layer)
        station_layer.add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    bounds = tracts.total_bounds
    map_view.fit_bounds([[float(bounds[1]), float(bounds[0])],
                         [float(bounds[3]), float(bounds[2])]])
    legend = f"""<div style="position:fixed;bottom:24px;left:16px;z-index:9999;
        background:white;padding:12px 14px;border:1px solid #aab7bf;border-radius:4px;
        max-width:345px;font:12px/1.45 Arial,sans-serif;color:#26333b;">
        <b style="font-size:14px">NYC Airbnb and subway access</b><br>
        <span style="color:{ORANGE}">●</span> {len(sample):,} of {len(locations):,} listings, seeded sample<br>
        Primary sample excludes hotel/hostel/resort listings. Nightly asking prices, October 2025.
        Coordinates may be displaced by about 150 m.
        Station distance is straight line, not a walking route. Station extract date unknown.<br>
        Optional income layer: ACS 2019–2023; missing-income tracts have no fill.<br>
        Sources: Inside Airbnb; MTA / NY Open Data; U.S. Census Bureau.
        Income and station proximity are not causal price estimates.</div>"""
    map_view.get_root().html.add_child(folium.Element(legend))
    destination = root / "results" / "nyc_airbnb_map.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(destination)
    return destination
