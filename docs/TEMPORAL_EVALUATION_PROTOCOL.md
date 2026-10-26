# Frozen temporal evaluation protocol, version 1

Written before downloading or opening the June and September listing files. Existing knowledge comprises the October 2025 analysis and source metadata/HEAD responses only. This is a pre-specified retrospective holdout evaluation, not a registered experiment or causal study. The protocol commit and its SHA must be recorded with the results.

## Application claim and population

Question: can an October-trained model provide a useful asking-price comparison for a host/listing unseen in the available historical snapshots, when contemporaneous property/location information is supplied later?

The intended use is screening and comparison of long-stay advertised prices. It is not automatic price posting, a forecast using only October information, a valuation, or an estimate of transactions, demand, revenue or profit.

Eligibility is fixed: finite positive advertised price; valid coordinates; unique listing ID; matched NYC 2023 tract; exclude room_type Hotel room or property_type containing hotel, hostel or resort (case-insensitive); advertised minimum_nights >=30. No price-tail trimming or test-specific price corrections. Missing-price records remain excluded and their counts/selection drift must be reported. Use the existing 2023 ACS/TIGER and frozen MTA references and the existing cleaning/features.

## Data roles

- Training: the original Inside Airbnb NYC 2025-10-01 archive, pinned in the existing source manifest.
- Calibration only: https://data.insideairbnb.com/united-states/ny/new-york-city/2026-06-14/data/listings.csv.gz . No parameter, feature or point-prediction fitting may use this snapshot's labels.
- Final holdout: https://data.insideairbnb.com/united-states/ny/new-york-city/2026-09-14/data/listings.csv.gz . No choices may be changed after its labels are opened. Source hashes, bytes and collection dates must be saved.

Novelty uses **all raw records**, including missing-price and ineligible rows. June's calibration cohort must have listing and host IDs absent from October's raw records. September's strict cohort must have listing and host IDs absent from both October and June raw records. This means unseen in these snapshots, not proven newly created on Airbnb.

For the primary estimand, select one eligible listing per novel host by the smallest SHA256 of `temporal-v1|` + listing ID, independent of prices. This host-uniform sample avoids large accounts dominating the evaluation. Score the all-eligible, novel-listing and all-strict-new-host-listing cohorts as separately labeled secondary populations. Never silently replace the primary cohort with an easier population.

Require >=300 calibration hosts and >=300 final primary hosts for an adoption decision; otherwise report insufficient independent sample. Do not lower this threshold after seeing results.

## Locked candidate models

Primary candidate: the existing histogram gradient-boosting pipeline **without transit**, trained only on eligible October rows, with per-row weights inverse to the October eligible listing count of its host, normalized to mean one. Host IDs themselves are never features. Host balancing is motivated by the previously observed account concentration and the host-uniform application target.

Use the existing feature pipeline, training-fold imputation and encoding, and fixed parameters: max_iter=140, learning_rate=0.07, max_leaf_nodes=15, min_samples_leaf=35, l2_regularization=1, early_stopping=False, random_state=42; target log1p(price), inverse expm1 with nonnegative prediction. Numeric features are existing BASE_NUMERIC + CONTEXT_NUMERIC; categorical features are room_type, borough, property_type. No tuning/search or test-based candidate selection.

Secondary candidates are (a) the same host-balanced model including the two existing transit features and (b) the original listing-weighted model without transit. They remain explicitly secondary; their performance must not silently replace a failed primary claim.

## Strong comparable-price baselines

Two baseline outputs use the same training-only hierarchy and support rules. One uses raw-dollar listing medians; the other uses the median of per-host raw-dollar medians in each cell. Both must be reported and the primary candidate must satisfy the superiority gate against **both**.

Keys: borough and recorded neighborhood strings; room_type; capacity exact 1..6 with >=7 grouped as 7+; bedrooms exact 0,1,2 with >=3 grouped as 3+; missing numeric keys Unknown. Minimum-stay bucket is exactly30, exactly31,32–59,60–89,90+. All normalization rules are training/test symmetric and independent of price.

Use the first supported cell in this fixed order:

1. borough, neighborhood, room_type, capacity, bedrooms, minimum-stay bucket
2. borough, neighborhood, room_type, capacity, bedrooms
3. borough, neighborhood, room_type, capacity
4. borough, room_type, capacity, bedrooms, minimum-stay bucket
5. borough, room_type, capacity, bedrooms
6. borough, room_type, capacity
7. borough, room_type
8. room_type
9. global

Each non-global cell needs >=20 training listings and >=5 distinct training hosts. Global fallback always exists. Save test fallback-level counts. Do not use future prices as comparables or add post-hoc hierarchy levels.

## Price intervals and metrics

For each frozen candidate/baseline, compute absolute log1p residuals on the host-uniform June calibration cohort. Nominal 80% interval radius is order statistic ceil((n_cal+1)*0.8), capped at the available maximum if needed. Inverse-transform [predicted_log-q,predicted_log+q], clip lower bounds at zero. No September recalibration.

Temporal drift and spatial/account dependence remove the usual exchangeability guarantee. Report empirical September coverage and width; do not claim distribution-free 80% future coverage.

For every labeled cohort report n/listing and host counts, MAE, RMSE, median absolute error, median absolute percentage error, share within25% of actual price, log1p R2, interval coverage and median interval width relative to predicted price+1. Keep units and populations explicit. Freeze all point predictions before reading September outcomes for scoring.

Paired uncertainty uses 3,000 bootstrap resamples of primary hosts with seed42. The primary cohort has one listing per host. Bootstrap the paired per-host loss vectors, not separate model samples. Evaluate relative MAE improvement against both strong baselines and relative MAE difference of transit vs no-transit. Use individual two-sided 98.3333% percentile intervals (Bonferroni allocation of a 5% family across these three contrasts). These are conditional empirical intervals, not guarantees over the entire housing market.

## Predeclared decisions

The primary model meets the chosen offline screening criteria only if all hold on the September primary cohort:

- the calibration/final sample gates are met;
- MAE is >=10% lower than **each** strong comparable baseline, with the corresponding simultaneous-family interval lower bound >0;
- median absolute percentage error <=30%;
- empirical nominal-80% interval coverage >=75%;
- median (upper-lower)/(point prediction+1) <=1.5.

These are researcher-chosen engineering targets, not an industry standard or demonstrated business impact. A failed condition must be reported as not meeting the criteria, not relabeled a success. Report subgroup failure rather than dropping that subgroup.

Practical predictive equivalence of adding transit is supported only if the entire predeclared interval for its relative MAE difference lies within [-2%,+2%]. Failing to demonstrate superiority is not evidence of equivalence. This never establishes a causal transit effect.

## Independent replication of the market benchmark

One principal descriptive comparison is fixed: Manhattan vs Brooklyn, Entire home/apt, bedrooms exactly1, capacity exactly2, minimum_nights exactly30, subway distance <=500m. Each borough/period needs >=100 listings and >=30 hosts. June and September must be reported regardless of direction. Report medians, the relative Brooklyn-minus-Manhattan gap, largest-host-exclusion and equal-host-median sensitivities, and account concentration. Capacity2–4, private rooms and800m remain labeled sensitivities.

A recurring benchmark claim requires Brooklyn's listing median to be >=10% lower in **both** future periods, and the direction to remain lower after excluding each borough's largest host. Otherwise report non-replication or insufficient sample. Do not call this a matched causal borough effect; amenities and neighborhood composition remain different. Do not use this market comparison to change model gates.

## Delivery and integrity

Store code, protocol SHA, source hashes, aggregate metrics, uncertainty, decision flags, failure analysis and a model card. Keep raw caches, host identities, listing-level predictions and serialized model objects out of Git. Publish an evidence-backed conclusion and a reproducible local quote tool only with its observed scope/limitations; any failed adoption gate must stay visible. Existing exploratory findings remain historical context rather than independent confirmation.
