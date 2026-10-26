# Executed findings

Results generated from the source hashes in `results/source_manifest.json`. Primary sample: October 2025 NYC listings excluding recorded hotel, hostel and resort products. All finite positive prices are retained. This is an observational portfolio study, not a deployment or causal evaluation.

## Sample and data quality

| Step | Listings |
| --- | ---: |
| Original source | 36,111 |
| Missing/invalid prices | 14,783 |
| Unique positive-price records | 21,328 |
| Outside/unmatched NYC tract | 2 |
| Priced spatial catalog | 21,326 |
| Hospitality category exclusions | 1,018 |
| Primary non-hotel sample | 20,308 |

The original catalog's hotel-room median is $40,000/night, with substantial masses at $40,000 and $50,000. Just 234 listings above $5,000, 1.10% of the catalog, account for 66.77% of summed asking prices. This sum is a tail diagnostic, not revenue. The source does not prove a cents conversion or justify changing those rates. Hospitality has a median minimum stay of 1 night, compared with 30 nights in the primary sample. Accommodation-product scope is preferable to an arbitrary price cap; hotel-labelled private rooms are also excluded. Remaining extreme residential asks are retained and reflected in RMSE.

## Predictive comparison

| Model | Random MAE ($/night) | Spatial MAE ($/night) | Spatial median AE ($) | Spatial log1p-price R² |
| --- | ---: | ---: | ---: | ---: |
| Global median | 141.36 | 141.55 | 75.00 | -0.006 |
| Room + borough median | 119.05 | 119.65 | 51.00 | 0.372 |
| Distance + income | 130.57 | 130.87 | 66.73 | 0.170 |
| Listing controls (ridge) | 103.95 | 104.90 | 45.41 | 0.548 |
| Full without transit | 75.38 | 79.22 | 31.32 | 0.742 |
| Full with transit | 75.18 | 79.34 | 31.72 | 0.743 |

Every row uses 20,308 held-out predictions, one per listing per validation design. All six candidates share the same folds. The strongest spatial MAE candidate, **Full without transit**, reduces error by **33.8%** against the room/borough median. Its median absolute error is **$31.32**, while RMSE **$545.23** reflects tail failures. R² is on log1p price, not raw dollars. Exponentiating a fitted log mean does not produce an unbiased expected dollar price.

Transit distance and station count add no spatial-MAE benefit beyond coordinates, property features and tract context. The tiny log-R² difference is not evidence of useful business lift. No confidence interval for the CV improvement is estimated; do not present the difference as a significance test. Hyperparameters are fixed; these internal candidate comparisons do not replace future-period, buffered, or host-group external validation.

The full catalog is a separate sensitivity population of 21,326 listings. Its full-with-transit spatial MAE is **$349.88**, compared with **$345.50** for the room/borough baseline, despite log-R² **0.792**. This illustrates why high log-R² alone is an inadequate headline. See `results/tables/full_catalog/model_metrics.csv` for all candidates.

## Does closer subway access independently predict higher asking prices?

The first three regressions use the same **14,001 complete cases across 1,757 tracts**. Per 500 m farther from subway, the distance/income association is **-3.07%**. Adding room type, borough, capacity, bedrooms, bathrooms, minimum stay, review count and rating reduces it to **-1.56%**. Adding neighborhood fixed effects yields **+0.80%**, with 95% tract-clustered interval **[-0.87%, 2.49%]**. Restricted-distance and central-price sensitivities also span zero.

These percentages transform a **natural-log-price** coefficient as `100 * (exp(beta * 0.5) - 1)`; prediction models separately use log1p price. This result supports uncertainty about the sign after these controls. It does not prove zero effect, rule out all associations, or estimate a causal station premium. Model choice, missing-data selection, a linear slope and unobserved listing quality remain limitations. Tract-clustered errors also do not remove correlation between neighboring tracts.

## Accessibility and spatial patterns

Median primary asking price is **$150/night** and median subway distance is **336 m**. **71.6%** fall within 500 m; shares at 350 m and 650 m are **52.5%** and **80.9%**. These are threshold-sensitivity bounds, not confidence intervals or a calibrated location-error model. Coordinates can be displaced up to about 150 m, so precise thresholds and boundary tract assignments should be treated cautiously. Subway distance is straight-line access to station coordinates, not entrances or walking routes.

Manhattan's median ask is **$214**, Brooklyn's **$130**, Queens' **$101**, Bronx's **$94**, and Staten Island's **$99**. These composition-dependent medians are descriptive. SIR is excluded, so Staten Island's large distance does not imply absence of rail transit.

Across 996 tracts with at least five primary listings, row-standardized symmetrized 8-nearest-neighbor Moran I is **0.661** for mean log1p prices and **0.074** for spatial OOF residuals of the **full-with-transit** candidate. Both two-sided permutation p-values are 0.002 with 499 seeded permutations. Tract positions are mean listing coordinates. These are exploratory diagnostics; residual clustering remains and warrants future spatial/host validation.

## What changed from the course version

Corrected projected-distance and map-coordinate errors; converted Census sentinel codes to missing; restricted statewide geometry to NYC; removed silent spatial-test failure; verified join cardinality; pinned and hashed input versions; replaced training-fit-only reporting with matched random/spatial CV and strong baselines; retained source anomalies and a full-catalog sensitivity; added executed figures, tables, notebook, provenance, and targeted automated checks. The saved original run records training R² 0.017678, while a historical slide claims 0.45. Neither is used as a new validated result.

## Next evidence needed

A second comparable listing snapshot would support temporal testing. Entrance/walking-network data would improve the access proxy. Host-group or buffered spatial validation would test stronger independence. Booking transactions would be required to evaluate achieved rates, occupancy or revenue. These remain future work, not implemented achievements.
