"""Exploratory market findings from explicit product/stay/access strata."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from .presentation import _style, BLUE, ORANGE, INK


def summarize_stay_access(root: Path) -> dict:
    root = Path(root)
    d = pd.read_csv(root / "data/processed/listings.csv")
    stay = pd.cut(d.minimum_nights, [-np.inf, 7, 29, np.inf],
                  labels=["Minimum 1–7 nights", "Minimum 8–29 nights", "Minimum ≥30 nights"])
    counts = stay.value_counts(sort=False).rename_axis("stay_requirement").reset_index(name="n")
    counts["share_pct"] = counts.n / len(d) * 100
    counts.to_csv(root / "results/tables/stay_requirement_shares.csv", index=False)
    d["distance_band"] = pd.cut(d.dist_km, [-np.inf, .5, 1, np.inf],
                                labels=["Within 500 m", "500–1,000 m", "Over 1,000 m"])
    longstay = d[d.minimum_nights >= 30]
    grouped = longstay.groupby(["room_type", "distance_band"], observed=True).agg(
        n=("price", "size"), median_price_usd=("price", "median")).reset_index()
    grouped.to_csv(root / "results/tables/long_stay_room_access.csv", index=False)
    return {"n": len(d), "minimum_stay_30plus_n": int((d.minimum_nights >= 30).sum()),
            "minimum_stay_30plus_pct": float((d.minimum_nights >= 30).mean() * 100),
            "minimum_stay_upto7_n": int((d.minimum_nights <= 7).sum()),
            "minimum_stay_upto7_pct": float((d.minimum_nights <= 7).mean() * 100),
            "scope": "October 2025 positive-priced non-hospitality sample. Minimum stay is a listing requirement, not an observed booking duration.",
            "long_stay_access": grouped.to_dict(orient="records")}


def render_insight_figures(root: Path):
    root = Path(root)
    _style()
    output = root / "figures"
    driver = pd.read_csv(root / "results/tables/grouped_permutation_importance.csv")
    driver = driver.sort_values("delta_mae_usd_median", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6.4))
    fig.subplots_adjust(left=.29, right=.95, bottom=.21, top=.77)
    colors = [ORANGE if v == "Subway accessibility" else BLUE for v in driver.group]
    y = np.arange(len(driver))
    delta = driver.delta_mae_usd_median.to_numpy()
    ax.barh(y, delta, color=colors, height=.62)
    ax.errorbar(delta, y, xerr=np.vstack([
        delta - driver.delta_mae_usd_min, driver.delta_mae_usd_max - delta]),
        fmt="none", ecolor=INK, capsize=3, linewidth=1)
    for position, value in enumerate(delta):
        ax.text(value + .7, position, f"+${value:.2f}", va="center", fontsize=11)
    ax.set_yticks(y, driver.group)
    ax.set_xlim(0, max(delta) * 1.18)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.set_xlabel("Increase in spatial held-out MAE after shuffling a feature group")
    ax.grid(axis="x", alpha=.2)
    fig.suptitle("Which information does the price model rely on?", x=.02, y=.96,
                 ha="left", fontsize=18, fontweight="bold")
    fig.text(.02, .885, f"Baseline MAE ${driver.baseline_mae_usd.iloc[0]:.2f}/night; same 20,308 spatial held-out predictions.",
             ha="left", fontsize=12)
    fig.text(.02, .035, "Bars: median across five pooled shuffle repeats. Whiskers: repeat min–max, not confidence intervals.\n"
             "Features in each group are shuffled jointly. Correlated groups share information; this is not a causal price effect.",
             fontsize=9, color="#52626c")
    fig.savefig(output / "price_information_groups.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    table = pd.read_csv(root / "results/tables/long_stay_room_access.csv")
    types = ["Entire home/apt", "Private room"]
    bands = ["Within 500 m", "500–1,000 m", "Over 1,000 m"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    fig.subplots_adjust(left=.1, right=.95, top=.75, bottom=.24)
    for room, color, marker in zip(types, [BLUE, ORANGE], ["o", "s"]):
        selected = table[table.room_type == room].set_index("distance_band").reindex(bands)
        ax.plot(np.arange(3), selected.median_price_usd, color=color, marker=marker, markersize=8,
                linewidth=2, label=room)
        for j, row in enumerate(selected.itertuples()):
            offset = 11 if room == "Entire home/apt" else -20
            horizontal = "left" if j == 0 else "right" if j == 2 else "center"
            x_offset = 8 if j == 0 else -8 if j == 2 else 0
            ax.annotate(f"${row.median_price_usd:,.0f} · n={row.n:,}",
                        (j, row.median_price_usd), xytext=(x_offset, offset), textcoords="offset points",
                        ha=horizontal, fontsize=10)
    ax.set_xticks(np.arange(3), bands)
    ax.set_ylim(0, table[table.room_type.isin(types)].median_price_usd.max() * 1.25)
    ax.set_ylabel("Median advertised nightly price (USD)")
    ax.set_xlabel("Straight-line distance to nearest subway station")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="upper right", frameon=False)
    fig.suptitle("The observed subway-price pattern differs by room type", x=.02, y=.96,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(.02, .88, "Listings requiring at least 30 nights; October 2025 non-hotel sample.", fontsize=12)
    fig.text(.02, .025, "Descriptive medians; location, capacity and quality composition differ across bands. No causal premium is estimated.\n"
             "Approximate listing coordinates; subway excludes SIR. Source: results/tables/long_stay_room_access.csv.",
             fontsize=9, color="#52626c")
    fig.savefig(output / "long_stay_access_patterns.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    borough = pd.read_csv(root / "results/tables/segment_borough.csv")
    selected = borough[(borough.stay_scope == "minimum_eq30") & (borough.maximum_station_distance_m == 500)]
    products = [("entire_1br_2guests", "Entire home: one bedroom, capacity 2"),
                ("private_1br_1to2guests", "Private room: one bedroom, capacity 1–2")]
    places = ["Manhattan", "Brooklyn", "Queens"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharex=True)
    fig.subplots_adjust(left=.1, right=.97, top=.75, bottom=.25, wspace=.34)
    for ax, (product, title) in zip(axes, products):
        rows = selected[selected.product_id == product].set_index("borough").reindex(places)
        rows = rows[rows.eligible_n100.eq(True)]
        if len(rows) < 2:
            raise ValueError("Each displayed product needs two boroughs with at least 100 listings")
        positions = np.arange(len(rows))
        ax.barh(positions, rows.median_price_usd, color=BLUE, height=.6)
        ax.set_yticks(positions, rows.index)
        ax.invert_yaxis()
        for i, row in enumerate(rows.itertuples()):
            ax.text(row.median_price_usd + 5, i, f"${row.median_price_usd:,.0f} · n={row.n:,}", va="center", fontsize=10)
        ax.set_xlim(0, 330)
        ax.set_title(title, fontsize=12, pad=15)
        ax.grid(axis="x", alpha=.2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:.0f}"))
        ax.set_xlabel("Median advertised nightly price")
    fig.suptitle("Borough price gaps vary across accommodation products", x=.02, y=.96,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(.02, .88, "Listed minimum exactly 30 nights; nearest subway ≤500 m; October 2025 non-hotel sample.", fontsize=12)
    fig.text(.02, .035, "Common product/stay/access conditions; amenities and exact neighborhood/quality composition still differ.\n"
             "Queens entire-home cell (n=97) is excluded from this view by the 100-listing minimum. No causal effect is estimated.\n"
             "Source: results/tables/segment_borough.csv. Separate tract-bootstrap intervals accompany the fixed comparisons.",
             fontsize=9, color="#52626c")
    fig.savefig(output / "comparable_borough_prices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_insights(root: Path, *, recompute=True) -> dict:
    root = Path(root)
    if recompute:
        from .driver_analysis import run_driver_analysis
        from .segment_analysis import run_segment_analysis
        from .host_analysis import run_host_analysis
        run_driver_analysis(root)
        run_segment_analysis(root)
        run_host_analysis(root)
    summary = summarize_stay_access(root)
    summary["exact30_comparisons"] = pd.read_csv(root / "results/tables/segment_borough_bootstrap.csv").query("eligible_n100 == True").to_dict(orient="records")
    summary["host_group_models"] = pd.read_csv(root / "results/tables/host_model_metrics.csv").to_dict(orient="records")
    summary["source_manifest"] = "results/source_manifest.json"
    render_insight_figures(root)
    summary["interpretation"] = "Exploratory extension using the existing historical sample. Comparisons are descriptive or model-reliance diagnostics; not causal effects or transaction prices."
    (root / "results/market_findings.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
