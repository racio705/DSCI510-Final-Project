# Frozen temporal validation

The locked primary model met every predeclared offline screening criterion. The September primary sample contains **311 host accounts**, one listing per host, unseen in all raw October and June records. The June-only calibration sample contains 1,159 accounts. The final sample is only 11 accounts above the declared 300-account floor; conclusions remain conditional on this historical sample.

The [protocol](TEMPORAL_EVALUATION_PROTOCOL.md) was committed before the future downloads as `b75e5a69daf625312ca2a4c582d3ddbf36657447`. This is a pre-specified retrospective holdout evaluation, not a registered experiment or a causal study.

## Decision and every locked criterion

| Criterion | Observed | Locked requirement | Result |
| --- | --- | --- | --- |
| Calibration and test hosts | 1,159 / 311 | Each ≥300 | PASS |
| Improvement over listing comparables | 16.3%; CI [7.80%, 24.87%] | Point improvement ≥10%; CI lower >0 | PASS |
| Improvement over host comparables | 13.6%; CI [5.33%, 22.21%] | Point improvement ≥10%; CI lower >0 | PASS |
| Median absolute percentage error | 23.5% | ≤30% | PASS |
| Observed nominal-80% interval coverage | 82.6% | ≥75% | PASS |
| Median relative interval width | 1.143 | ≤1.5 | PASS |

These are researcher-chosen engineering targets for offline screening. They are not industry certification or demonstrated commercial returns. Width means `(upper − lower)/(point prediction + 1)`; its median dollar width is $168.65. Temporal drift removes the usual exchangeability guarantee for the nominal 80% intervals.

![Temporal holdout performance](../figures/temporal_holdout.png)

## Candidate and comparable-price benchmarks

| Candidate | MAE/night | Median AE | Median APE | Within ±25% | 80% coverage | Relative width |
| --- | --- | --- | --- | --- | --- | --- |
| Primary: host-balanced HGB without transit | $96.03 | $32.42 | 23.5% | 53.1% | 82.6% | 1.143 |
| Secondary: host-balanced HGB with transit | $96.69 | $33.15 | 22.8% | 51.8% | 82.6% | 1.127 |
| Secondary: listing-weighted HGB without transit | $95.94 | $32.48 | 24.3% | 50.8% | 84.2% | 1.104 |
| Baseline: comparable listing medians | $114.73 | $40.41 | 27.0% | 46.6% | 83.9% | 1.516 |
| Baseline: comparable host medians | $111.09 | $37.57 | 27.1% | 46.9% | 83.0% | 1.420 |

The two strong baselines use October-only comparable cells, falling back through a fixed hierarchy when a cell has fewer than 20 training listings or five hosts. One computes listing medians, the other medians of within-cell host medians. The primary HGB is trained on 17,100 October long-stay listings from 8,339 hosts, with weights inverse to each host's eligible listing count. Host IDs are not features. The original listing-weighted candidate remains secondary even if a metric is numerically better.

The primary MAE is $96.03/night, median absolute error $32.42, median absolute percentage error 23.5%, and 53.1% of predictions are within 25% of the observed asking price. This is a screening benchmark with material individual uncertainty, not automatic price posting.

## Paired uncertainty and transit

| Contrast | Point estimate | 98.3333% paired-bootstrap interval |
| --- | --- | --- |
| Primary MAE improvement vs comparable_listing_median | 16.30% | [7.80%, 24.87%] |
| Primary MAE improvement vs comparable_host_median | 13.56% | [5.33%, 22.21%] |
| Transit vs no-transit relative MAE change | 0.68% | [-0.89%, 2.41%] |

There are 3,000 paired resamples of the same primary hosts with seed 42. The individual 98.3333% percentile intervals allocate a 5% Bonferroni family across the two baseline improvements and the transit contrast. Positive baseline contrasts favor the primary model; positive transit changes mean higher error with transit. These are conditional empirical intervals.

Practical predictive equivalence was not established: the interval does not fit inside the predeclared ±2% margin. A lack of superiority is not evidence of equivalence. The transit point difference is 0.68%, interval [-0.89%, 2.41%]. Neither result identifies a causal transit effect.

## Recurring market benchmark

The fixed descriptive benchmark met its recurrence criteria in both future snapshots. The locked principal comparison is entire homes, one bedroom, declared capacity two, minimum exactly 30 nights and subway distance ≤500 m. Each borough/period requires at least 100 listings and 30 hosts. Brooklyn must be at least 10% lower in both 2026 periods and remain lower after excluding each borough's largest host.

| Period | Manhattan median | Brooklyn median | BK gap | Gap after largest-host exclusion | Status |
| --- | --- | --- | --- | --- | --- |
| 2025-10-01 | $220.00 (n=1,193; hosts=564) | $159.00 (n=561; hosts=500) | -27.73% | -15.14% | historical context |
| 2026-06-14 | $251.56 (n=1,229; hosts=550) | $161.13 (n=539; hosts=485) | -35.95% | -22.34% | meets period criteria |
| 2026-09-14 | $240.57 (n=1,032; hosts=482) | $163.76 (n=449; hosts=397) | -31.93% | -18.43% | meets period criteria |

![Fixed market benchmark](../figures/temporal_market_replication.png)

| Period | Borough | Largest-account listings | Account share | Median after exclusion | Median of host medians |
| --- | --- | --- | --- | --- | --- |
| 2025-10-01 | Manhattan | 267 | 22.4% | $185.00 | $195.00 |
| 2025-10-01 | Brooklyn | 9 | 1.6% | $157.00 | $158.25 |
| 2026-06-14 | Manhattan | 301 | 24.5% | $206.06 | $209.79 |
| 2026-06-14 | Brooklyn | 9 | 1.7% | $160.03 | $159.48 |
| 2026-09-14 | Manhattan | 267 | 25.9% | $199.13 | $203.54 |
| 2026-09-14 | Brooklyn | 9 | 2.0% | $162.44 | $162.40 |

October is historical context and does not count as a future replication. Market rows use all eligible listings in the fixed category, so listings and accounts can recur across periods; this is separate from the strict novel-host prediction test. Shared category filters do not match apartment floor area, quality, amenities or neighborhood composition. Capacity 2–4, private-room and 800 m rows remain labeled sensitivities in the source CSV and cannot rescue a failed principal comparison.

## Cohort and subgroup boundaries

| Cohort: primary model only | Listings | Hosts | MAE | Median APE | 80% coverage |
| --- | --- | --- | --- | --- | --- |
| All eligible September listings | 14152 | 7407 | $88.25 | 21.8% | 85.6% |
| September listings absent from all prior raw IDs | 1010 | 595 | $86.60 | 22.2% | 85.2% |
| All strict novel-host and novel-listing rows | 398 | 311 | $89.58 | 22.0% | 83.2% |
| PRIMARY one listing per strict novel host | 311 | 311 | $96.03 | 23.5% | 82.6% |

The primary cohort selects one strict novel-host listing using the smallest SHA256 of `temporal-v1|` plus listing ID, independent of prices. Novelty checks use all prior raw IDs, including missing-price and ineligible records. “Unseen” means absent from those snapshots, not proven newly created on Airbnb. September does not recalibrate intervals or refit point models.

| Borough | Room type | n | Primary MAE | Median APE | 80% coverage | Diagnostic flags |
| --- | --- | --- | --- | --- | --- | --- |
| Bronx | Entire home/apt | 5 | $95.12 | 9.7% | 80.0% | n<30; MAE not better than ≥1 baseline |
| Bronx | Private room | 7 | $26.40 | 25.2% | 85.7% | n<30; MAE not better than ≥1 baseline |
| Brooklyn | Entire home/apt | 86 | $90.88 | 16.6% | 84.9% | No listed diagnostic flag |
| Brooklyn | Private room | 35 | $60.32 | 24.9% | 85.7% | No listed diagnostic flag |
| Brooklyn | Shared room | 2 | $114.74 | 29.6% | 50.0% | n<30; coverage<75% |
| Manhattan | Entire home/apt | 77 | $168.30 | 24.4% | 76.6% | No listed diagnostic flag |
| Manhattan | Private room | 24 | $52.73 | 14.7% | 87.5% | n<30 |
| Queens | Entire home/apt | 33 | $58.36 | 27.8% | 87.9% | No listed diagnostic flag |
| Queens | Private room | 38 | $72.07 | 32.9% | 78.9% | median APE>30% |
| Queens | Shared room | 2 | $37.77 | 13.3% | 100.0% | n<30 |
| Staten Island | Entire home/apt | 1 | $70.65 | 34.1% | 100.0% | n<30; median APE>30% |
| Staten Island | Private room | 1 | $14.18 | 32.8% | 100.0% | n<30; MAE not better than ≥1 baseline; median APE>30% |

Subgroup flags are diagnostics, not extra post-hoc adoption gates. Rows with n<30 are shown but do not support reliable subgroup claims. Worse baseline comparisons or weak coverage are retained rather than dropping the affected group. Full model/baseline subgroup metrics are in `results/tables/temporal_holdout_subgroups.csv`.

## Selection, integrity and permitted use

| Snapshot | Raw rows | Missing/invalid price | Raw missing-price rate | Eligible long-stay rows | Eligible rows missing income |
| --- | --- | --- | --- | --- | --- |
| 2025-10-01 | 36,111 | 14,783 | 40.9% | 17,100 | 259 |
| 2026-06-14 | 30,259 | 8,744 | 28.9% | 16,380 | 217 |
| 2026-09-14 | 29,918 | 10,512 | 35.1% | 14,152 | 189 |

Missing-price rates use all raw rows as the denominator. The observed positive-price population is not all NYC accommodation. Eligibility excludes recorded hotel/hostel/resort categories and minimum stays below 30 nights, with no price-tail trimming. Coordinates can be displaced by about 150 m; the station extract date is unknown. ACS 2019–2023 and TIGER/Line 2023 references remain frozen.

The intended use is historical long-stay asking-price screening and comparison using contemporaneous 2026 property/location information. It is not a forecast using only October information, a booked-price estimate, a property valuation, or evidence of demand, occupancy, revenue, legal eligibility or profit.

Source hashes: `results/temporal_source_manifest.json`. Prediction freeze: `results/temporal_prediction_freeze.json`. Complete decisions/metrics: `results/temporal_validation.json`. Primary candidate and both baselines were fixed before future scoring; June labels calibrate intervals only. No secondary candidate replaces a failed primary decision.
