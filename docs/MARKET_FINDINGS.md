> The final locked temporal validation and runnable quote tool are documented in [Temporal validation](TEMPORAL_VALIDATION.md). This page preserves the earlier single-snapshot analysis.

# Usable market findings

These are exploratory findings from the **October 2025, positive-priced non-hotel sample**, not current booking offers or causal estimates. The source has missing prices, category uncertainty and approximate locations. The analysis separates advertised minimum stay, accommodation product, declared capacity and subway proximity before interpreting borough medians.

## 1. Most of the modeled sample has long-stay requirements

**17,100 of 20,308 listings (84.2%) require at least 30 nights.** Only 3,174 (15.6%) have a minimum of seven nights or fewer. These are advertised requirements, not observed booking durations. The appropriate application is a segmented asking-price benchmark, not an unrestricted tourist-night budget estimate. Tables: `segment_scope.csv` and `segment_minimum_nights_thresholds.csv`.

## 2. A borough price gap remains within a narrower product category

For **entire homes, one bedroom, declared capacity exactly two, minimum exactly 30 nights, and subway distance ≤500 m**, the listing-weighted median is:

| Borough | Listings | Median advertised nightly price |
| --- | ---: | ---: |
| Manhattan | 1,193 | $220 |
| Brooklyn | 561 | $159 |

Brooklyn is **$61, or 27.7%, lower** in this category. The exploratory 95% tract-bootstrap interval for Brooklyn minus Manhattan is **[-$84, -$40]**, or **[-35.6%, -20.0%]**. Resampling retains within-tract dependence but not dependence between neighboring tracts or from the same host account operating across tracts; intervals are pointwise and not adjusted for multiple comparisons. These are common category filters, not matching on amenities, quality, precise neighborhood or apartment floor area.

For private rooms with one bedroom, capacity one or two, the same exact-30-night and subway filters give **$79 in Manhattan (n=1,111), $65 in Brooklyn (n=1,563), and $67 in Queens (n=504)**. Thus a citywide borough-price multiplier is a poor substitute for product-specific benchmarks. The exact-capacity entire-home Queens cell has n=97, below the declared 100-listing reporting threshold; no interval is reported. Capacity 2–4 and distance ≤800 m appear separately as sensitivities. Tables: `segment_borough.csv`, `segment_borough_bootstrap.csv`.

## 3. Host-account composition materially changes the benchmark

Among listings whose minimum is exactly 31 nights, **895 of 1,149 (77.9%) belong to one host account**. The group's median asking price is $395; after excluding that account, the remaining 254 listings have median $140. This is an account-composition comparison, not evidence that changing a minimum from 30 to 31 nights raises prices. A host ID is a platform account, not a verified person or company.

In the earlier broad minimum-≥30 entire-home category, mixing these 31-night listings raised Manhattan's median to $310 and produced a 48.4% Brooklyn gap. Fixing the minimum at exactly 30 nights lowers Manhattan's median to $220 and the gap to 27.7%. **The 48.4% figure is not used as a primary matched-product result.** The broader category remains in `segment_borough_bootstrap_sensitivity.csv`.

Even exact-30 comparisons are composition-sensitive: removing each borough's largest host gives **Manhattan $185 versus Brooklyn $157**, a **15.1%** gap. Giving each host one median instead of weighting every listing gives $195 versus $158.25. Exclusion and host weighting change the population/estimand; they are sensitivities, not a corrected causal price premium. The direction persists, while the magnitude is not a universal discount. Aggregate tables contain no host identifiers: `host_concentration.csv` and `host_composition_relationship.csv`.

## 4. Size, product type and geographic context carry more predictive information than transit

The fixed full-with-transit candidate has spatial OOF MAE **$79.34/night**. Jointly shuffling held-out capacity information increases error by **$38.82**, geography/tract context by **$29.31**, and room/property type by **$22.32**. Shuffling subway distance/count increases MAE by only **$0.38**. These are median error increases across five pooled shuffle repeats using the same five spatial folds. They are model-reliance diagnostics, not explained-price percentages, directional effects or intervention estimates; correlated groups share information and shuffles can create unrealistic feature combinations.

Unadjusted long-stay medians also differ by product: private rooms are **$70, $69 and $70** at ≤500 m, 500–1,000 m and >1,000 m, respectively. Entire-home medians are $213, $195 and $138. These bands have different borough and size compositions, so they do not establish a causal transit premium. Together with neighborhood fixed-effects and the transit ablation, they argue against applying one uniform subway markup. Tables: `grouped_permutation_importance.csv`, `long_stay_room_access.csv`.

## 5. The model retains a useful advantage when holding out whole host accounts

Supplemental five-fold host-group evaluation keeps all listings from a host in one fold. For **10,227 distinct host accounts and 20,308 held-out predictions**, the without-transit candidate achieves **$90.49 MAE**, compared with **$122.99** for the room/borough median baseline, a **26.4% reduction**. Adding transit yields $90.87 and does not improve MAE. This tests same-snapshot transfer to unseen host accounts. It is separate from spatial CV and is neither a combined spatial/host test nor future-period external validation. Table: `host_model_metrics.csv`.

## How to use these findings

Build asking-price comparison groups in this order: accommodation product and capacity, exact minimum-stay requirement, location/subway constraints, then host-account concentration and listing quality. Publish listing-weighted and host-sensitive summaries when account concentration is material. The findings support segmentation, comparison and model validation; they do not establish achieved booking prices, occupancy, revenue, legal eligibility or a realized commercial return.

Reproduce after the main pipeline with `python src/derive_findings.py`. The main pipeline now includes this extension. The same historical data informed exploration and refinement, so all new intervals and subgroup comparisons remain exploratory. Sources and exact hashes are documented in `docs/DATA_SOURCES.md` and `results/source_manifest.json`.
