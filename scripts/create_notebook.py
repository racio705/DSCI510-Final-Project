"""Build a portable, offline analytical notebook from a completed local run.

The generated notebook is deliberately unexecuted. Execute it top-to-bottom
with nbclient or Jupyter, then retain its populated outputs for review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


def create_notebook(root: Path | str) -> Path:
    root = Path(root).resolve()
    markdown = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = [
        markdown("""# NYC Airbnb prices and subway access

An extension of the original DSCI 510 project using its October 2025 listing snapshot, MTA subway station locations, and ACS 2019–2023 tract context. The primary analysis excludes hospitality listings using the recorded room and property types; a separate full-catalog analysis retains them.

This notebook reads the pipeline's saved local inputs and results. It makes no network calls. It distinguishes descriptive price differences, adjusted observational associations, and out-of-fold prediction performance. Full source definitions and limitations are in [Data sources](../docs/DATA_SOURCES.md)."""),
        markdown("## Load the saved analysis"),
        code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown

candidates = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in candidates if (p / 'results' / 'metrics.json').is_file()
             and (p / 'src' / 'nyc_airbnb').is_dir()), None)
if ROOT is None:
    raise FileNotFoundError('Run this notebook from the repository or notebooks directory after running the pipeline.')

metrics = json.loads((ROOT / 'results' / 'metrics.json').read_text(encoding='utf-8'))
quality = metrics.get('quality', metrics)
model_metrics = pd.read_csv(ROOT / 'results' / 'tables' / 'model_metrics.csv')
coefficients = pd.read_csv(ROOT / 'results' / 'tables' / 'coefficient_estimates.csv')
boroughs = pd.read_csv(ROOT / 'results' / 'tables' / 'borough_summary.csv')
segments = pd.read_csv(ROOT / 'results' / 'tables' / 'distance_segments.csv')
listings = pd.read_csv(ROOT / 'data' / 'processed' / 'listings.csv',
                      usecols=['id', 'price', 'latitude', 'longitude', 'dist_km', 'median_household_income'],
                      dtype={'id': str})
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 12)
"""),
        markdown("## Results at a glance"),
        code("""within_500 = (listings['dist_km'] <= 0.5).mean() * 100
summary = [
    f"**{len(listings):,} positive-priced listings in the primary sample**, after NYC tract matching and the stated hospitality exclusion.",
    f"Median advertised price: **${listings['price'].median():,.0f}/night**.",
    f"**{within_500:.1f}%** are within 500 m of a subway station in straight-line distance.",
]
spatial_rows = model_metrics[model_metrics['validation'].astype(str).str.contains('spatial', case=False)]
if not spatial_rows.empty:
    best = spatial_rows.sort_values('mae_usd').iloc[0]
    summary.append(f"Lowest spatial-validation MAE among the evaluated candidates: "
                   f"**${best['mae_usd']:,.2f}/night** ({str(best['model']).replace('_', ' ')}).")
if isinstance(quality.get('raw_listings'), (int, float)):
    summary.append(f"The source contained {int(quality['raw_listings']):,} listing rows; "
                   "missing/invalid prices and spatial eligibility affect the usable sample.")
display(Markdown((chr(10) * 2).join(summary)))
"""),
        markdown("""## Context and methods

The outcome is the advertised nightly listing price. The primary population excludes records with room type `Hotel room` or property types matching `hotel|hostel|resort` (case-insensitive). This uses recorded listing taxonomy, rather than verifying a property's legal status. No price-tail trimming is applied to the primary population. Subway distances use a local metric projection and station coordinates; Staten Island Railway is excluded. Census tract income and population are lagged area characteristics.

Prediction models are compared using random and spatial-block out-of-fold predictions. Preprocessing is fitted within each training fold. Dollar errors and log1p-price R² answer different questions, so both are retained. The separate inference regressions use ln(price), report their own sample sizes and confidence intervals, and estimate conditional associations.

### Assumptions and scope

- The historical snapshot describes available listing records, rather than a census of completed stays.
- Positive-priced eligible records define this analysis population. Missing prices can create selection bias.
- A 500 m straight-line threshold is an access proxy. Listing location anonymization and station entrance locations limit its precision.
- Spatial validation holds out the stated geographic groups. It does not demonstrate future-period or host-independent performance.
- Property, neighborhood, and host factors can confound subway-price associations. No coefficient is interpreted as a causal transit effect."""),
        markdown("## Prediction performance in the primary sample"),
        code("""display(Image(filename=str(ROOT / 'figures' / 'model_validation.png')))
display(model_metrics.round({'mae_usd': 2, 'rmse_usd': 2, 'median_ae_usd': 2, 'r2_log': 3}))
"""),
        markdown("""Dollar MAE is the average absolute nightly-price error across held-out predictions. RMSE gives larger errors more weight. Median absolute error describes the typical listing. Compare models on the same validation population; a random/spatial gap measures sensitivity to the partition, rather than an automatic deployment guarantee."""),
        markdown("""### Full-catalog sensitivity

The source includes hospitality records with advertised prices near $40,000–50,000 per night. These are suspected data/price anomalies, not verified luxury-market observations. They are retained without correction in a separate full-catalog benchmark so the population choice remains inspectable. Its errors should not be compared as if measured on the primary sample."""),
        code("""full_catalog_path = ROOT / 'results' / 'tables' / 'full_catalog' / 'model_metrics.csv'
if full_catalog_path.exists():
    full_catalog = pd.read_csv(full_catalog_path)
    display(full_catalog.round({'mae_usd': 2, 'rmse_usd': 2, 'median_ae_usd': 2, 'r2_log': 3}))
else:
    display(Markdown('Full-catalog sensitivity table is unavailable in this run; do not infer its performance.'))
"""),
        markdown("## Subway distance after controls"),
        code("""display(Image(filename=str(ROOT / 'figures' / 'subway_association.png')))
columns = ['specification', 'n', 'tract_clusters', 'percent_per_500m', 'ci_lower_pct', 'ci_upper_pct']
display(coefficients[columns].round(2))
"""),
        markdown("""The inference regressions use ln(price). The percentage scale is `100 × (exp(0.5 × coefficient_per_km) − 1)`: the modeled relative price difference for 500 m farther from a station, holding the included controls constant. This transformation describes a geometric-price association, not a dollar change in the arithmetic mean. Intervals use tract-clustered regression uncertainty. The first three specifications use the same complete-case sample, so their differences reflect controls rather than a changing sample. An interval spanning zero supports uncertainty about the sign under that specification."""),
        markdown("## Price differences within room types"),
        code("""display(Image(filename=str(ROOT / 'figures' / 'distance_by_room_type.png')))
display(segments.round({'median_price': 2}).head(20))
"""),
        markdown("""These medians describe the primary population. The number of listings is displayed because groups can have very different sample sizes. Room-type separation addresses one major source of mixing, while amenities and neighborhood differences remain. All panels share a logarithmic price axis, and retained price tails are not silently clipped."""),
        markdown("## Geographic coverage and accessibility"),
        code("""display(Image(filename=str(ROOT / 'figures' / 'listing_geography.png')))
display(boroughs.round({'median_price': 2, 'within_500m_pct': 1, 'median_distance_m': 0}))
"""),
        markdown("""[Open the interactive geographic map](../results/nyc_airbnb_map.html) to inspect the seeded listing sample and optional subway/tract-income layers. Map popups use the original WGS84 latitude and longitude. Base tiles and interactive-map JavaScript require internet access; all four figures above remain portable notebook image outputs.

Staten Island's large distance to the subway should be interpreted with the explicit exclusion of Staten Island Railway. Low station distance alone does not measure access to jobs, service frequency, entrances, or travel time."""),
        markdown("## Data checks and frozen input hashes"),
        code("""assert listings['id'].is_unique, 'Listing IDs must be unique'
assert np.isfinite(listings['price']).all() and (listings['price'] > 0).all()
assert listings['latitude'].between(-90, 90).all()
assert listings['longitude'].between(-180, 180).all()
assert np.isfinite(listings['dist_km']).all() and (listings['dist_km'] >= 0).all()
assert model_metrics['n'].eq(len(listings)).all(), 'Model metrics and primary listing population differ'
if isinstance(quality.get('primary_listing_count'), (int, float)):
    assert len(listings) == int(quality['primary_listing_count'])
elif isinstance(quality.get('analysis_listings'), (int, float)):
    assert len(listings) == int(quality['analysis_listings'])
assert (coefficients['ci_lower_pct'] <= coefficients['percent_per_500m']).all()
assert (coefficients['ci_upper_pct'] >= coefficients['percent_per_500m']).all()

quality_keys = ['raw_listings', 'missing_or_invalid_price', 'nonpositive_price',
                'duplicate_listing_ids_removed', 'unmatched_nyc_tract_listings',
                'priced_spatial_listings', 'hospitality_excluded', 'primary_listing_count',
                'analysis_listings', 'listings_missing_income', 'unique_station_locations']
checks = [{'check': key.replace('_', ' '), 'value': quality[key]}
          for key in quality_keys if key in quality]
display(pd.DataFrame(checks))

manifest = quality.get('source_manifest', {})
if not manifest:
    manifest_path = ROOT / 'data' / 'raw' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
sources = manifest.get('sources', {})
hashes = [{'source': key, 'filename': record.get('filename'), 'sha256': record.get('sha256')}
          for key, record in sources.items()]
display(pd.DataFrame(hashes))
print('Notebook arithmetic and input checks passed; source definitions remain in docs/DATA_SOURCES.md.')
"""),
        markdown("## Written interpretation"),
        code("""interpretation = ROOT / 'docs' / 'RESULTS.md'
if interpretation.exists():
    display(Markdown(interpretation.read_text(encoding='utf-8')))
else:
    display(Markdown('Read the executed tables and intervals above together. Predictive evaluation and '
                     'conditional association are separate results; neither establishes a causal subway premium.'))
"""),
        markdown("""## Limitations and reproduction

The listing snapshot is from October 2025; ACS measures 2019–2023, and the MTA cache's original download date is unknown. Inside Airbnb reports approximate coordinate displacement up to 150 m. ACS estimates have sampling uncertainty and some capped/missing income estimates. Missing-price listings are outside the modeled population. Unavailability does not establish occupancy or revenue.

Rerun the repository's documented pipeline before executing this notebook if changing inputs, features, splits, or models. Keep the frozen manifest with each run. This notebook uses saved results and is a review companion, rather than an alternative model-training implementation. See [README](../README.md), [data provenance](../docs/DATA_SOURCES.md), and [Inside Airbnb's assumptions](https://insideairbnb.com/data-assumptions/)."""),
    ]
    insertion = next(i for i, cell in enumerate(cells) if cell.cell_type == "markdown" and cell.source == "## Written interpretation")
    cells[insertion:insertion] = [
        markdown("""## Market findings from explicit accommodation segments

The following exploratory extension separates advertised stay requirements, compares fixed product/access categories, and measures model reliance on feature groups. It uses the same historical sample, rather than an independent confirmation dataset. Minimum nights is a listing rule, not an observed booking duration. Product categories do not match amenities or exact neighborhood quality."""),
        code("""findings = ROOT / 'docs' / 'MARKET_FINDINGS.md'
display(Markdown(findings.read_text(encoding='utf-8')))
"""),
        code("""display(Image(filename=str(ROOT / 'figures' / 'comparable_borough_prices.png')))
comparison = pd.read_csv(ROOT / 'results' / 'tables' / 'segment_borough_bootstrap.csv')
comparison = comparison[comparison['eligible_n100'].eq(True)]
display(comparison[['product_id', 'comparison_borough', 'n_reference', 'n_comparison',
                    'pct_difference', 'ci95_lower_pct', 'ci95_upper_pct']].round(2))
"""),
        markdown("""The displayed intervals are exploratory, pointwise 95% percentile intervals from 1,000 census-tract bootstrap draws within each borough. Primary comparisons hold the advertised minimum at exactly 30 nights. They retain within-tract dependence but do not account for cross-tract dependence or adjust for multiple comparisons. They are not a causal estimate of changing a listing's borough."""),
        code("""display(Image(filename=str(ROOT / 'figures' / 'price_information_groups.png')))
importance = pd.read_csv(ROOT / 'results' / 'tables' / 'grouped_permutation_importance.csv')
display(importance[['group', 'baseline_mae_usd', 'delta_mae_usd_median',
                    'delta_log_rmse_median']].round(3))
host_metrics = pd.read_csv(ROOT / 'results' / 'tables' / 'host_model_metrics.csv')
display(host_metrics.round({'mae_usd': 2, 'rmse_usd': 2, 'median_ae_usd': 2, 'r2_log': 3}))
display(Image(filename=str(ROOT / 'figures' / 'long_stay_access_patterns.png')))
"""),
        markdown("""Feature groups are shuffled jointly in held-out spatial folds of the full-with-transit candidate. Importance is conditional on the other correlated groups and can include unrealistic shuffled combinations. It measures predictive reliance, not how much an intervention would change prices. Shuffle repeat ranges are not confidence intervals. Distance-band medians are unadjusted descriptive summaries."""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    })
    nbf.validate(notebook)
    destination = root / "notebooks" / "portfolio_results.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    print(create_notebook(arguments.root))


if __name__ == "__main__":
    main()
